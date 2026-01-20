#!/bin/bash
# 빠른 샘플 테스트 (10개 샘플만)

set -e

# ============================================================================
# 설정
# ============================================================================

MODEL_PATH="./checkpoints/shapellm-7bs-scaffold-scaffold_i3ce-lorascaffold-i3ce_251119_1759"
MODEL_BASE="qizekun/ShapeLLM_7B_gapartnet_v1.0"
DATA_DIR="./playground/data/shapellm/scaffold_sft_color"
OUTPUT_DIR="./quick_test_results"
mkdir -p $OUTPUT_DIR

# ============================================================================
# Quick Test: 10 samples only
# ============================================================================

echo "🚀 Quick Test: 10 Validation Samples"
echo "========================================"

python scripts/batch_inference.py \
  --model_path $MODEL_PATH \
  --model_base $MODEL_BASE \
  --data_dir $DATA_DIR \
  --split val \
  --output $OUTPUT_DIR/predictions_quick.json \
  --sample_points_num 10000 \
  --with_color \
  --max_samples 10

echo ""
echo "✅ Quick test complete!"
echo "📁 Results: $OUTPUT_DIR/predictions_quick.json"
echo ""
echo "📊 Sample predictions:"
python -c "
import json
preds = json.load(open('$OUTPUT_DIR/predictions_quick.json'))
for i, p in enumerate(preds[:3], 1):
    print(f'\n--- Sample {i}: {p[\"id\"]} ---')
    print(f'Q: {p[\"conversations\"][0][\"value\"][:80]}...')
    print(f'A: {p[\"conversations\"][1][\"value\"][:150]}...')
"
