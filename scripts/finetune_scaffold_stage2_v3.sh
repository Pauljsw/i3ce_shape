#!/bin/bash
# ==============================================================================
# Stage 2 V3: Instruction Tuning for Scaffold Missing Detection
# ==============================================================================
#
# Changes in V3:
#   - Uses V3 data (no text shortcuts, diverse questions)
#   - 5 epochs instead of 1
#   - Optional: Fine-tuned ReCon++ encoder
#   - Better learning rate scheduling
#
# Environment: 4x RTX 3090 (24GB each)
# ==============================================================================

set -e

# Configuration
LLM_VERSION="qizekun/ShapeLLM_7B_gapartnet_v1.0"
OUTPUT_NAME="scaffold-stage2-v3-lora"

# Stage 1 checkpoint
STAGE1_CHECKPOINT="./checkpoints/scaffold-stage1-v3"
MM_PROJECTOR_PATH="${STAGE1_CHECKPOINT}/mm_projector.bin"

# V3 Data paths
DATA_PATH="./playground/data/shapellm/scaffold_v3/stage2_sft_train.json"
PCS_PATH="./playground/data/shapellm/scaffold_v3/pcs"

# ReCon++ encoder options:
# Option 1: Original pretrained (default)
VISION_TOWER_PATH="./checkpoints/recon/large.pth"

# Option 2: Fine-tuned on scaffold (uncomment to use)
# VISION_TOWER_PATH="./checkpoints/recon_scaffold/ckpt-best.pth"

# Output directory
OUTPUT_DIR="./checkpoints/${OUTPUT_NAME}"
LOG_DIR="./logs/${OUTPUT_NAME}"

mkdir -p $LOG_DIR

echo "=============================================================="
echo "Stage 2 V3: Instruction Tuning (5 epochs)"
echo "=============================================================="
echo "Stage 1: ${STAGE1_CHECKPOINT}"
echo "Data: ${DATA_PATH}"
echo "Vision Encoder: ${VISION_TOWER_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: 5"
echo "LoRA: r=128, alpha=256"
echo "=============================================================="

# Check prerequisites
if [ ! -d "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at ${STAGE1_CHECKPOINT}"
    echo "Please run Stage 1 V3 training first:"
    echo "  bash scripts/pretrain_scaffold_stage1_v3.sh"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Training data not found at ${DATA_PATH}"
    echo "Please generate V3 data first:"
    echo "  python -m tools.scaffold_data_pipeline.main \\"
    echo "      --num-scenes 2000 \\"
    echo "      --output-dir ./playground/data/shapellm/scaffold_v3"
    exit 1
fi

# Find mm_projector.bin
if [ -f "$MM_PROJECTOR_PATH" ]; then
    echo "Using mm_projector from: ${MM_PROJECTOR_PATH}"
elif [ -f "${STAGE1_CHECKPOINT}/non_lora_trainables.bin" ]; then
    MM_PROJECTOR_PATH="${STAGE1_CHECKPOINT}/non_lora_trainables.bin"
    echo "Using mm_projector from non_lora_trainables.bin"
else
    # Find latest checkpoint with mm_projector
    LATEST=$(ls -t ${STAGE1_CHECKPOINT}/checkpoint-*/mm_projector.bin 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        MM_PROJECTOR_PATH="$LATEST"
        echo "Using mm_projector from: ${MM_PROJECTOR_PATH}"
    else
        echo "ERROR: Cannot find mm_projector.bin"
        echo "Looking for non_lora_trainables.bin instead..."
        LATEST=$(ls -t ${STAGE1_CHECKPOINT}/checkpoint-*/non_lora_trainables.bin 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            MM_PROJECTOR_PATH="$LATEST"
            echo "Using: ${MM_PROJECTOR_PATH}"
        else
            echo "ERROR: Cannot find mm_projector weights"
            exit 1
        fi
    fi
fi

# Run training with 5 epochs
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --model_name_or_path $LLM_VERSION \
    --version v1 \
    --data_path $DATA_PATH \
    --point_folder $PCS_PATH \
    --vision_tower ReConV2/cfgs/pretrain/large/openshape.yaml \
    --vision_tower_path $VISION_TOWER_PATH \
    --pretrain_mm_mlp_adapter $MM_PROJECTOR_PATH \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --freeze_mm_mlp_adapter False \
    --mm_vision_select_layer -2 \
    --mm_use_pt_start_end False \
    --mm_use_pt_patch_token False \
    --sample_points_num 10000 \
    --with_color True \
    --with_ape True \
    --with_local True \
    --with_global True \
    --prompt_token_num 32 \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --logging_dir $LOG_DIR

echo "=============================================================="
echo "Stage 2 V3 Training Complete!"
echo "=============================================================="
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Next step: Run evaluation with:"
echo "  python -m llava.eval.model_vqa \\"
echo "    --model-path ${OUTPUT_DIR} \\"
echo "    --model-base ${LLM_VERSION} \\"
echo "    --question-file ./playground/data/shapellm/scaffold_v3/test_questions.jsonl \\"
echo "    --point-folder ./playground/data/shapellm/scaffold_v3/pcs \\"
echo "    --answers-file ${OUTPUT_DIR}/test_answers.jsonl \\"
echo "    --conv-mode scaffold_missing \\"
echo "    --temperature 0 \\"
echo "    --num_beams 1"
echo "=============================================================="
