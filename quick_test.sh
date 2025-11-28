#!/bin/bash
# Quick test of optimized training (5 epochs each phase)
# This should complete in ~30 minutes and achieve val_loss < 0.50

cd $SCRATCH/mheight_project

python train_optimized.py \
    --model_size baseline \
    --pretrain_epochs 5 \
    --finetune_epochs 5 \
    --ds1_input data/DS-1-samples_n_k_m_P.pkl \
    --ds1_output data/DS-1-samples_mHeights.pkl \
    --ds2_input data/DS-2-samples_n_k_m_P.pkl \
    --ds2_output data/DS-2-samples_mHeights.pkl \
    --ds3_input data/DS-3-Train-n_k_m_P.pkl \
    --ds3_output data/DS-3-Train-mHeights.pkl \
    --pretrain_ckpt checkpoints/test_pretrain.weights.h5 \
    --final_ckpt checkpoints/test_final.weights.h5

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "=========================================="
echo ""
echo "Check the last few lines for final val_loss."
echo "If val_loss < 0.50, you're good to run full training!"
echo ""
echo "For full training, run:"
echo "  python train_optimized.py --model_size baseline --pretrain_epochs 80 --finetune_epochs 100"
echo ""
