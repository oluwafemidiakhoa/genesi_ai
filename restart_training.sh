#!/bin/bash
# Restart training with fixed NaN loss issues

echo "🔄 Restarting Genesis RNA training with fixes..."
echo ""
echo "Changes applied:"
echo "  ✅ Fixed NaN loss when no structure/pair annotations"
echo "  ✅ Set structure_loss_weight = 0.0"
echo "  ✅ Set pair_loss_weight = 0.0"
echo "  ✅ Training focuses on MLM task only"
echo ""
echo "Starting training in 3 seconds..."
sleep 3

cd /home/user/genesi_ai

python examples/train_with_ncrna.py \
    --data_path data/human_ncrna/sequences.pkl \
    --model_size base \
    --batch_size 16 \
    --num_epochs 10 \
    --use_ast \
    --ast_target_activation 0.4 \
    --output_dir checkpoints/pretrained/base
