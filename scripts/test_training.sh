#!/bin/bash
# Quick test training to verify improvements

echo "=========================================="
echo "Testing Genesis RNA Training Improvements"
echo "=========================================="
echo ""

# Create test output directory
TEST_DIR="./test_checkpoints"
mkdir -p "$TEST_DIR"

echo "Running 3-epoch test training with all improvements enabled..."
echo "Using T4-optimized configuration"
echo ""

# Run training with config
cd /home/user/genesi_ai

python -m genesis_rna.train_pretrain \
    --config configs/train_t4_optimized.yaml \
    --output_dir "$TEST_DIR" \
    --num_epochs 3 \
    --batch_size 16 \
    --use_dummy_data

echo ""
echo "=========================================="
echo "Test Training Complete!"
echo "=========================================="
echo ""

# Check if metrics file exists
if [ -f "$TEST_DIR/training_metrics.csv" ]; then
    echo "✅ Metrics logging: WORKING"
    echo "   Metrics saved to: $TEST_DIR/training_metrics.csv"
    echo ""

    # Show first few lines
    echo "First few metric entries:"
    head -n 5 "$TEST_DIR/training_metrics.csv"
    echo ""

    # Run visualization
    echo "Generating visualizations..."
    python scripts/visualize_metrics.py \
        --metrics_file "$TEST_DIR/training_metrics.csv" \
        --output_dir "$TEST_DIR"

    echo ""
else
    echo "⚠️  Metrics file not found!"
fi

# Check if checkpoint exists
if [ -f "$TEST_DIR/best_model.pt" ]; then
    echo "✅ Model checkpointing: WORKING"
    echo ""
fi

echo "Test results saved to: $TEST_DIR/"
echo ""
echo "To run full training on Google Colab with T4:"
echo "  python -m genesis_rna.train_pretrain \\"
echo "    --config configs/train_t4_optimized.yaml \\"
echo "    --output_dir /content/drive/MyDrive/genesis_rna_checkpoints \\"
echo "    --data_path ./data/rnacentral_processed \\"
echo "    --num_epochs 10"
echo ""
