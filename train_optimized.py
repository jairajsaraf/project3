#!/usr/bin/env python3
"""
Optimized m-Height Transformer Training (TensorFlow)
Based on PyTorch baseline that achieved 0.374 val_loss

Key improvements:
- 20x augmentation (not 3x)
- OneCycleLR scheduler with cosine annealing
- L1 regularization on weights
- Proper pretrain/finetune strategy
- Two model sizes: baseline (128) and large (256)

Usage:
    python train_optimized.py --mode pretrain --model_size baseline
    python train_optimized.py --mode pretrain --model_size large
"""

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedShuffleSplit

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Enable mixed precision
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)
logger.info("✓ Mixed precision enabled")


# ============================================================================
# CONFIGURATION
# ============================================================================

class BaseConfig:
    """Baseline config matching PyTorch (DMODEL=128)"""
    # Problem constants
    N_FIXED = 9
    K_VALUES = [4, 5, 6]
    M_VALUES = [2, 3, 4, 5, 6]
    MAX_K = 6
    MAX_PARITY = 5
    
    # Model architecture (matches PyTorch baseline)
    DMODEL = 128
    NHEAD = 4
    ENC_FF = 384
    ENC_LAYERS = 3
    DROPOUT = 0.10
    
    # Training hyperparameters (from PyTorch)
    BATCH_SIZE = 128
    TRAIN_AUG_FACTOR = 20  # ← 20x augmentation like PyTorch!
    VAL_SPLIT = 0.20
    
    # Learning rate (OneCycleLR)
    MAX_LR = 2e-3  # Peak learning rate
    START_LR_FACTOR = 10.0  # Start LR = MAX_LR / 10
    FINAL_LR_FACTOR = 1e3   # Final LR = MAX_LR / 1e3
    PCT_START = 0.15        # 15% warmup
    
    # Regularization
    WEIGHT_DECAY = 2e-4  # L2 via AdamW
    L1_COEFF = 1e-6      # L1 on weights
    CLIPNORM = 1.0
    
    # Early stopping
    PATIENCE = 20


class LargeConfig(BaseConfig):
    """Large config for beating baseline (DMODEL=256)"""
    DMODEL = 256
    NHEAD = 8
    ENC_FF = 1024
    ENC_LAYERS = 6
    DROPOUT = 0.15
    
    # Slightly lower LR for stability
    MAX_LR = 1.5e-3
    
    # More aggressive regularization
    L1_COEFF = 2e-6


# ============================================================================
# ONECYCLE LEARNING RATE SCHEDULER
# ============================================================================

class OneCycleLR(keras.callbacks.Callback):
    """
    OneCycleLR scheduler matching PyTorch implementation.
    
    Phase 1 (warmup): LR increases from start_lr to max_lr
    Phase 2 (anneal): LR decreases from max_lr to final_lr using cosine
    """
    
    def __init__(self, max_lr, steps_per_epoch, epochs, 
                 pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = steps_per_epoch * epochs
        self.step_count = 0
        
        # Calculate LR boundaries
        self.start_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        # Calculate step boundaries
        self.warmup_steps = int(self.total_steps * pct_start)
        self.anneal_steps = self.total_steps - self.warmup_steps
        
        logger.info(f"OneCycleLR: {self.start_lr:.2e} → {self.max_lr:.2e} → {self.final_lr:.2e}")
        logger.info(f"  Warmup: {self.warmup_steps} steps, Anneal: {self.anneal_steps} steps")
    
    def on_train_batch_begin(self, batch, logs=None):
        if self.step_count < self.warmup_steps:
            # Phase 1: Linear warmup
            progress = self.step_count / self.warmup_steps
            lr = self.start_lr + (self.max_lr - self.start_lr) * progress
        else:
            # Phase 2: Cosine annealing
            progress = (self.step_count - self.warmup_steps) / self.anneal_steps
            lr = self.final_lr + (self.max_lr - self.final_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))
        
        self.model.optimizer.learning_rate.assign(lr)
        self.step_count += 1


# ============================================================================
# L1 REGULARIZATION CALLBACK
# ============================================================================

class L1Regularization(keras.callbacks.Callback):
    """
    Add L1 penalty on weights (excluding biases and layer norms).
    Applied after each batch like in PyTorch.
    """
    
    def __init__(self, l1_coeff):
        super().__init__()
        self.l1_coeff = l1_coeff
    
    def on_train_batch_end(self, batch, logs=None):
        if self.l1_coeff <= 0:
            return
        
        l1_loss = 0.0
        for layer in self.model.layers:
            for weight in layer.trainable_weights:
                # Skip biases and normalization layers
                if 'bias' in weight.name.lower() or 'norm' in weight.name.lower():
                    continue
                l1_loss += tf.reduce_sum(tf.abs(weight))
        
        # Add L1 loss to the total loss
        if logs and 'loss' in logs:
            logs['loss'] += self.l1_coeff * l1_loss


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_sample(P, k, n, max_k):
    """
    Apply sign flips and column permutations.
    P: (k, n-k) array
    Returns: (max_k, n-k) padded array
    """
    k_int = int(k)
    cols = n - k_int
    
    # Random sign flips on each row
    signs = np.random.choice([-1, 1], size=k_int)
    P_flipped = P * signs[:, np.newaxis]
    
    # Random column permutation
    perm = np.random.permutation(cols)
    P_aug = P_flipped[:, perm]
    
    # Pad to max_k rows
    P_padded = np.zeros((max_k, cols), dtype=np.float32)
    P_padded[:k_int, :] = P_aug
    
    return P_padded


# ============================================================================
# DATASET CREATION
# ============================================================================

def create_dataset(n_vals, k_vals, m_vals, P_matrices, mheights, config, augment=True):
    """
    Create infinite TensorFlow dataset with proper augmentation.
    
    CRITICAL: Generator has infinite loop (while True) to prevent data exhaustion.
    NO .repeat() call on the dataset itself - that causes memory issues.
    """
    import numpy as np
    import tensorflow as tf

    def generator():
        # INFINITE LOOP - dataset never exhausts
        while True:
            indices = np.arange(len(n_vals))
            np.random.shuffle(indices)

            for idx in indices:
                n, k, m = n_vals[idx], k_vals[idx], m_vals[idx]
                P = P_matrices[idx]
                mheight = mheights[idx]

                aug_factor = config.TRAIN_AUG_FACTOR if augment else 1

                for _ in range(aug_factor):
                    if augment:
                        P_aug = augment_sample(P, k, n, config.MAX_K)
                    else:
                        cols = n - k
                        P_padded = np.zeros((config.MAX_K, cols), dtype=np.float32)
                        P_padded[:k, :] = P
                        P_aug = P_padded

                    # tokens: (seq_len, max_k) where seq_len = n - k
                    tokens = P_aug.T.astype(np.float32)
                    seq_len = tokens.shape[0]

                    # mask: (seq_len,)
                    mask = np.ones(seq_len, dtype=bool)

                    # categorical indices
                    k_idx = config.K_VALUES.index(int(k))
                    m_idx = config.M_VALUES.index(int(m))

                    # parity feature
                    parity_idx = min(abs(int(P.sum())) // 10, config.MAX_PARITY)

                    # target in log2 space
                    target = np.log2(mheight + 1.0).astype(np.float32)

                    yield (tokens, mask, k_idx, m_idx, parity_idx, target)

    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, config.MAX_K), dtype=tf.float32),  # tokens
            tf.TensorSpec(shape=(None,), dtype=tf.bool),                  # mask
            tf.TensorSpec(shape=(), dtype=tf.int32),                      # k_idx
            tf.TensorSpec(shape=(), dtype=tf.int32),                      # m_idx
            tf.TensorSpec(shape=(), dtype=tf.int32),                      # parity_idx
            tf.TensorSpec(shape=(), dtype=tf.float32),                    # target
        ),
    )

    def prepare_batch(tokens, mask, k_idx, m_idx, parity_idx, target):
        inputs = {
            "tokens": tokens,
            "mask": mask,
            "k_idx": k_idx,
            "m_idx": m_idx,
            "parity_idx": parity_idx,
        }
        return inputs, target

    dataset = dataset.map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)

    # Padded batch for variable sequence lengths
    padded_shapes = (
        {
            "tokens": tf.TensorShape([None, config.MAX_K]),
            "mask": tf.TensorShape([None]),
            "k_idx": tf.TensorShape([]),
            "m_idx": tf.TensorShape([]),
            "parity_idx": tf.TensorShape([]),
        },
        tf.TensorShape([]),
    )

    padding_values = (
        {
            "tokens": tf.cast(0.0, tf.float32),
            "mask": False,
            "k_idx": tf.cast(0, tf.int32),
            "m_idx": tf.cast(0, tf.int32),
            "parity_idx": tf.cast(0, tf.int32),
        },
        tf.cast(0.0, tf.float32),
    )

    dataset = dataset.padded_batch(
        config.BATCH_SIZE,
        padded_shapes=padded_shapes,
        padding_values=padding_values,
        drop_remainder=False,
    )

    # NO .repeat() - generator is already infinite!
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SinusoidalPositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, :, :])
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        # Cast PE to match input dtype (handles mixed precision)
        pe = tf.cast(self.pe[:, :seq_len, :], x.dtype)
        return x + pe


class MultiHeadPooling(layers.Layer):
    """Multi-head pooling combining mean, max, min, and attention."""
    
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.attn_weights = layers.Dense(1, use_bias=False)
    
    def call(self, x, mask=None):
        # x: (batch, seq, d_model)
        # mask: (batch, seq)

        if mask is not None:
            # Cast mask to float32 for stable division
            mask_float = tf.cast(mask, tf.float32)[:, :, tf.newaxis]
            # Cast x to float32 for mean calculation
            x_float32 = tf.cast(x, tf.float32)
            masked_x = x_float32 * mask_float
            # Sum of mask values (number of valid tokens per batch)
            seq_len = tf.reduce_sum(mask_float, axis=1, keepdims=False) + 1e-6  # Shape: [batch, 1]
        else:
            x_float32 = tf.cast(x, tf.float32)
            masked_x = x_float32
            # Ensure consistent shape [batch, 1] for seq_len
            batch_size = tf.shape(x)[0]
            seq_len = tf.fill([batch_size, 1], tf.cast(tf.shape(x)[1], tf.float32))

        # Mean pooling in float32 for numerical stability
        mean_pool = tf.reduce_sum(masked_x, axis=1) / seq_len  # Shape: [batch, d_model]
        mean_pool = tf.cast(mean_pool, x.dtype)  # Cast back to mixed precision dtype

        # Max pooling - use -1e4 instead of -1e9 to avoid overflow in float16
        if mask is not None:
            masked_for_max = tf.where(mask[:, :, tf.newaxis], x, tf.cast(-1e4, x.dtype))
        else:
            masked_for_max = x
        max_pool = tf.reduce_max(masked_for_max, axis=1)

        # Min pooling - use 1e4 instead of 1e9 to avoid overflow in float16
        if mask is not None:
            masked_for_min = tf.where(mask[:, :, tf.newaxis], x, tf.cast(1e4, x.dtype))
        else:
            masked_for_min = x
        min_pool = tf.reduce_min(masked_for_min, axis=1)

        # Attention pooling - use -1e4 instead of -1e9
        attn_scores = self.attn_weights(x)  # (batch, seq, 1)
        if mask is not None:
            attn_scores = tf.where(mask[:, :, tf.newaxis], attn_scores, tf.cast(-1e4, x.dtype))
        attn_weights = tf.nn.softmax(attn_scores, axis=1)
        attn_pool = tf.reduce_sum(x * attn_weights, axis=1)

        # Concatenate all pooling methods
        pooled = tf.concat([mean_pool, max_pool, min_pool, attn_pool], axis=-1)
        return pooled


def create_model(config):
    """Create m-Height Transformer model."""
    
    # Inputs
    tokens_input = layers.Input(shape=(None, config.MAX_K), name='tokens')
    mask_input = layers.Input(shape=(None,), dtype=tf.bool, name='mask')
    k_idx_input = layers.Input(shape=(), dtype=tf.int32, name='k_idx')
    m_idx_input = layers.Input(shape=(), dtype=tf.int32, name='m_idx')
    parity_idx_input = layers.Input(shape=(), dtype=tf.int32, name='parity_idx')
    
    # Token embedding
    x = layers.Dense(config.DMODEL, name='token_embed')(tokens_input)
    
    # Positional encoding
    x = SinusoidalPositionalEncoding(config.DMODEL)(x)
    
    # Categorical embeddings
    k_embed = layers.Embedding(len(config.K_VALUES), config.DMODEL // 4, name='k_embed')(k_idx_input)
    m_embed = layers.Embedding(len(config.M_VALUES), config.DMODEL // 4, name='m_embed')(m_idx_input)
    parity_embed = layers.Embedding(config.MAX_PARITY + 1, config.DMODEL // 4, name='parity_embed')(parity_idx_input)
    
    # Transformer encoder
    for i in range(config.ENC_LAYERS):
        # Multi-head attention
        attn_out = layers.MultiHeadAttention(
            num_heads=config.NHEAD,
            key_dim=config.DMODEL // config.NHEAD,
            dropout=config.DROPOUT,
            name=f'mha_{i}'
        )(x, x, attention_mask=mask_input[:, tf.newaxis, tf.newaxis, :])
        
        x = layers.LayerNormalization(epsilon=1e-6, name=f'ln1_{i}')(x + attn_out)
        
        # Feedforward
        ff_out = layers.Dense(config.ENC_FF, activation='relu', name=f'ff1_{i}')(x)
        ff_out = layers.Dropout(config.DROPOUT, name=f'drop_{i}')(ff_out)
        ff_out = layers.Dense(config.DMODEL, name=f'ff2_{i}')(ff_out)
        
        x = layers.LayerNormalization(epsilon=1e-6, name=f'ln2_{i}')(x + ff_out)
    
    # Multi-head pooling
    pooled = MultiHeadPooling(config.DMODEL, name='pooling')(x, mask_input)
    
    # Concatenate with categorical embeddings
    combined = layers.Concatenate(name='concat')([pooled, k_embed, m_embed, parity_embed])
    
    # Prediction head
    out = layers.Dense(config.DMODEL, activation='relu', name='head1')(combined)
    out = layers.Dropout(config.DROPOUT, name='head_drop')(out)
    out = layers.Dense(1, dtype='float32', name='output')(out)  # Force float32 for output
    
    model = keras.Model(
        inputs=[tokens_input, mask_input, k_idx_input, m_idx_input, parity_idx_input],
        outputs=out,
        name='mheight_transformer'
    )
    
    return model


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(input_path, output_path):
    """Load dataset from pickle files.
    
    Input format: List of samples, each: [n, k, m, P_matrix]
    Output format: List of mheights
    """
    with open(input_path, 'rb') as f:
        samples = pickle.load(f)
    with open(output_path, 'rb') as f:
        mheights = pickle.load(f)
    
    # Extract n, k, m, P arrays from list of samples
    n_vals = []
    k_vals = []
    m_vals = []
    P_matrices = []
    
    for sample in samples:
        n, k, m, P = sample
        n_vals.append(n)
        k_vals.append(k)
        m_vals.append(m)
        P_matrices.append(P)
    
    # Convert to numpy arrays
    n_vals = np.array(n_vals)
    k_vals = np.array(k_vals)
    m_vals = np.array(m_vals)
    P_matrices = np.array(P_matrices, dtype=object)  # Object array for variable-size matrices
    mheights = np.array(mheights)
    
    return n_vals, k_vals, m_vals, P_matrices, mheights


# ============================================================================
# TRAINING
# ============================================================================

def train_pretrain_mode(config, args):
    """
    Pretrain mode: Train on DS-1+DS-2, then finetune on DS-3.
    This matches the PyTorch "combined ALL" strategy.
    """
    
    logger.info("=" * 70)
    logger.info("PRETRAIN MODE")
    logger.info("=" * 70)
    
    # Step 1: Pretrain on DS-1 + DS-2
    logger.info("STEP 1: PRETRAINING ON DS-1 + DS-2")
    logger.info("=" * 70)
    
    ds1_input = args.ds1_input or 'data/DS-1-samples_n_k_m_P.pkl'
    ds1_output = args.ds1_output or 'data/DS-1-samples_mHeights.pkl'
    ds2_input = args.ds2_input or 'data/DS-2-Train-n_k_m_P.pkl'
    ds2_output = args.ds2_output or 'data/DS-2-Train-mHeights.pkl'
    
    logger.info(f"Loading DS-1 from {ds1_input}...")
    n1, k1, m1, P1, mh1 = load_dataset(ds1_input, ds1_output)
    logger.info(f"DS-1: {len(n1)} samples")
    
    logger.info(f"Loading DS-2 from {ds2_input}...")
    n2, k2, m2, P2, mh2 = load_dataset(ds2_input, ds2_output)
    logger.info(f"DS-2: {len(n2)} samples")
    
    # Combine
    n_combined = np.concatenate([n1, n2])
    k_combined = np.concatenate([k1, k2])
    m_combined = np.concatenate([m1, m2])
    P_combined = np.concatenate([P1, P2])
    mh_combined = np.concatenate([mh1, mh2])
    
    logger.info(f"Combined: {len(n_combined)} samples")
    logger.info(f"With {config.TRAIN_AUG_FACTOR}x augmentation = {len(n_combined) * config.TRAIN_AUG_FACTOR} training samples")
    
    # Split train/val
    strat_key = [f"{k}_{m}" for k, m in zip(k_combined, m_combined)]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.VAL_SPLIT, random_state=42)
    train_idx, val_idx = next(splitter.split(P_combined, strat_key))
    
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Create datasets
    train_ds = create_dataset(
        n_combined[train_idx], k_combined[train_idx], m_combined[train_idx],
        P_combined[train_idx], mh_combined[train_idx],
        config, augment=True
    )
    
    val_ds = create_dataset(
        n_combined[val_idx], k_combined[val_idx], m_combined[val_idx],
        P_combined[val_idx], mh_combined[val_idx],
        config, augment=False
    )
    
    # Create model
    model = create_model(config)
    
    # Compile
    optimizer = keras.optimizers.AdamW(
        learning_rate=config.MAX_LR / config.START_LR_FACTOR,  # Start LR
        weight_decay=config.WEIGHT_DECAY,
        clipnorm=config.CLIPNORM
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Simple MSE in log2 space
        metrics=['mse']
    )
    
    # Calculate steps
    train_samples = len(train_idx) * config.TRAIN_AUG_FACTOR
    val_samples = len(val_idx)
    steps_per_epoch = train_samples // config.BATCH_SIZE
    validation_steps = val_samples // config.BATCH_SIZE
    
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")
    
    # Callbacks
    pretrain_epochs = args.pretrain_epochs
    
    callbacks = [
        OneCycleLR(
            max_lr=config.MAX_LR,
            steps_per_epoch=steps_per_epoch,
            epochs=pretrain_epochs,
            pct_start=config.PCT_START,
            div_factor=config.START_LR_FACTOR,
            final_div_factor=config.FINAL_LR_FACTOR
        ),
        L1Regularization(config.L1_COEFF),
        keras.callbacks.ModelCheckpoint(
            args.pretrain_ckpt,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
    ]
    
    # Train
    logger.info(f"Training for {pretrain_epochs} epochs...")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=pretrain_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2
    )
    
    logger.info("Pretraining complete!")
    logger.info(f"Best val_loss: {min(history.history['val_loss']):.6f}")
    
    # Step 2: Finetune on DS-3
    logger.info("=" * 70)
    logger.info("STEP 2: FINETUNING ON DS-3")
    logger.info("=" * 70)
    
    ds3_input = args.ds3_input or 'data/DS-3-Train-n_k_m_P.pkl'
    ds3_output = args.ds3_output or 'data/DS-3-Train-mHeights.pkl'
    
    logger.info(f"Loading DS-3 from {ds3_input}...")
    n3, k3, m3, P3, mh3 = load_dataset(ds3_input, ds3_output)
    logger.info(f"DS-3: {len(n3)} samples")
    
    # Split
    strat_key3 = [f"{k}_{m}" for k, m in zip(k3, m3)]
    splitter3 = StratifiedShuffleSplit(n_splits=1, test_size=config.VAL_SPLIT, random_state=42)
    train_idx3, val_idx3 = next(splitter3.split(P3, strat_key3))
    
    train_ds3 = create_dataset(
        n3[train_idx3], k3[train_idx3], m3[train_idx3],
        P3[train_idx3], mh3[train_idx3],
        config, augment=True
    )
    
    val_ds3 = create_dataset(
        n3[val_idx3], k3[val_idx3], m3[val_idx3],
        P3[val_idx3], mh3[val_idx3],
        config, augment=False
    )
    
    # Calculate steps for finetune
    train_samples3 = len(train_idx3) * config.TRAIN_AUG_FACTOR
    val_samples3 = len(val_idx3)
    steps_per_epoch3 = train_samples3 // config.BATCH_SIZE
    validation_steps3 = val_samples3 // config.BATCH_SIZE
    
    finetune_epochs = args.finetune_epochs
    
    # New OneCycleLR for finetuning
    callbacks3 = [
        OneCycleLR(
            max_lr=config.MAX_LR * 0.5,  # Lower LR for finetuning
            steps_per_epoch=steps_per_epoch3,
            epochs=finetune_epochs,
            pct_start=config.PCT_START,
            div_factor=config.START_LR_FACTOR,
            final_div_factor=config.FINAL_LR_FACTOR
        ),
        L1Regularization(config.L1_COEFF),
        keras.callbacks.ModelCheckpoint(
            args.final_ckpt,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
    ]
    
    logger.info(f"Finetuning for {finetune_epochs} epochs...")
    
    history2 = model.fit(
        train_ds3,
        validation_data=val_ds3,
        epochs=finetune_epochs,
        steps_per_epoch=steps_per_epoch3,
        validation_steps=validation_steps3,
        callbacks=callbacks3,
        verbose=2
    )
    
    logger.info("Finetuning complete!")
    logger.info(f"Final val_loss: {min(history2.history['val_loss']):.6f}")
    
    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Optimized m-Height Transformer Training')
    
    # Mode
    parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain'],
                        help='Training mode (only pretrain supported)')
    
    # Model size
    parser.add_argument('--model_size', type=str, default='baseline', choices=['baseline', 'large'],
                        help='Model size: baseline (DMODEL=128) or large (DMODEL=256)')
    
    # Data paths
    parser.add_argument('--ds1_input', type=str, default=None)
    parser.add_argument('--ds1_output', type=str, default=None)
    parser.add_argument('--ds2_input', type=str, default=None)
    parser.add_argument('--ds2_output', type=str, default=None)
    parser.add_argument('--ds3_input', type=str, default=None)
    parser.add_argument('--ds3_output', type=str, default=None)
    
    # Checkpoints
    parser.add_argument('--pretrain_ckpt', type=str, default='checkpoints/pretrain_best.weights.h5')
    parser.add_argument('--final_ckpt', type=str, default='checkpoints/final_best.weights.h5')
    
    # Epochs
    parser.add_argument('--pretrain_epochs', type=int, default=80)
    parser.add_argument('--finetune_epochs', type=int, default=100)
    
    args = parser.parse_args()
    
    # Select config
    if args.model_size == 'baseline':
        config = BaseConfig()
        logger.info("Using BASELINE config (DMODEL=128, matching PyTorch)")
    else:
        config = LargeConfig()
        logger.info("Using LARGE config (DMODEL=256, beating PyTorch)")
    
    logger.info(f"Model: DMODEL={config.DMODEL}, NHEAD={config.NHEAD}, "
                f"ENC_FF={config.ENC_FF}, ENC_LAYERS={config.ENC_LAYERS}")
    logger.info(f"Augmentation: {config.TRAIN_AUG_FACTOR}x")
    logger.info(f"Learning rate: {config.MAX_LR:.2e}")
    
    # Create checkpoints directory
    os.makedirs(os.path.dirname(args.pretrain_ckpt), exist_ok=True)
    os.makedirs(os.path.dirname(args.final_ckpt), exist_ok=True)
    
    # Train
    if args.mode == 'pretrain':
        model = train_pretrain_mode(config, args)
    
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
