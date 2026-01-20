#!/bin/bash
# 전체 평가 파이프라인 자동화 스크립트

set -e  # Exit on error

# ============================================================================
# 설정 (여기를 수정하세요!)
# ============================================================================

# 모델 경로
MODEL_PATH="./checkpoints/shapellm-7bs-scaffold-scaffold_i3ce-lorascaffold-i3ce_251119_1759"
MODEL_BASE="qizekun/ShapeLLM_7B_gapartnet_v1.0"

# 데이터 경로
DATA_DIR="./playground/data/shapellm/scaffold_sft_color"

# 출력 경로
OUTPUT_DIR="./evaluation_results"
mkdir -p $OUTPUT_DIR

# Point cloud 설정 (훈련 세팅과 동일하게!)
SAMPLE_POINTS=10000
WITH_COLOR="--with_color"  # color 사용 시, 없으면 제거

# ============================================================================
# Step 1: Validation Set 추론
# ============================================================================

echo "========================================"
echo "🚀 Step 1: Validation Set Inference"
echo "========================================"

PRED_VAL="$OUTPUT_DIR/predictions_val.json"

python scripts/batch_inference.py \
  --model_path $MODEL_PATH \
  --model_base $MODEL_BASE \
  --data_dir $DATA_DIR \
  --split val \
  --output $PRED_VAL \
  --sample_points_num $SAMPLE_POINTS \
  $WITH_COLOR

echo "✅ Validation predictions saved to $PRED_VAL"

# ============================================================================
# Step 2: Validation Set 평가
# ============================================================================

echo ""
echo "========================================"
echo "📊 Step 2: Validation Set Evaluation"
echo "========================================"

RESULTS_VAL="$OUTPUT_DIR/results_val.json"

python scripts/evaluate_scaffold.py \
  --predictions $PRED_VAL \
  --data_dir $DATA_DIR \
  --output $RESULTS_VAL

echo "✅ Validation results saved to $RESULTS_VAL"

# ============================================================================
# Step 3: Test Set 추론 (선택사항)
# ============================================================================

read -p "🤔 Run test set evaluation? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "========================================"
    echo "🚀 Step 3: Test Set Inference"
    echo "========================================"

    PRED_TEST="$OUTPUT_DIR/predictions_test.json"

    python scripts/batch_inference.py \
      --model_path $MODEL_PATH \
      --model_base $MODEL_BASE \
      --data_dir $DATA_DIR \
      --split test \
      --output $PRED_TEST \
      --sample_points_num $SAMPLE_POINTS \
      $WITH_COLOR

    echo "✅ Test predictions saved to $PRED_TEST"

    # ============================================================================
    # Step 4: Test Set 평가
    # ============================================================================

    echo ""
    echo "========================================"
    echo "📊 Step 4: Test Set Evaluation"
    echo "========================================"

    RESULTS_TEST="$OUTPUT_DIR/results_test.json"

    python scripts/evaluate_scaffold.py \
      --predictions $PRED_TEST \
      --data_dir $DATA_DIR \
      --output $RESULTS_TEST

    echo "✅ Test results saved to $RESULTS_TEST"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================"
echo "✅ Evaluation Complete!"
echo "========================================"
echo "Results directory: $OUTPUT_DIR"
ls -lh $OUTPUT_DIR/
echo ""
echo "📊 View results:"
echo "   cat $RESULTS_VAL"

if [[ -f "$RESULTS_TEST" ]]; then
    echo "   cat $RESULTS_TEST"
fi
