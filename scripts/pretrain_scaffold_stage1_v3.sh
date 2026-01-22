#!/bin/bash
# ==============================================================================
# Stage 1 V3: Feature Alignment for Scaffold Missing Detection
# ==============================================================================
#
# Changes in V3:
#   - Uses V3 data (no text shortcuts)
#   - More epochs (10 instead of 5)
#   - Optional: Use fine-tuned ReCon++ encoder
#   - Larger effective batch size for better convergence
#
# Environment: 4x RTX 3090 (24GB each)
# ==============================================================================

set -e

# Configuration
LLM_VERSION="qizekun/ShapeLLM_7B_gapartnet_v1.0"
OUTPUT_NAME="scaffold-stage1-v3"

# V3 Data paths (generated via scaffold_data_pipeline)
# Stage 1 uses CAPTION data for feature alignment (NOT QA data)
DATA_PATH="./playground/data/shapellm/scaffold_v3/stage1_caption_train.json"
PCS_PATH="./playground/data/shapellm/scaffold_v3/pcs"

# To generate V3 data:
# python -m tools.scaffold_data_pipeline.main \
#     --num-scenes 2000 \
#     --output-dir ./playground/data/shapellm/scaffold_v3

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
echo "Stage 1 V3: Feature Alignment Training"
echo "=============================================================="
echo "Data: ${DATA_PATH}"
echo "Vision Encoder: ${VISION_TOWER_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: 10"
echo "Batch: 2 x 8 x 4GPU = 64 effective"
echo "=============================================================="

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Stage 1 caption data not found at ${DATA_PATH}"
    echo "Please generate V3 data first:"
    echo "  python -m tools.scaffold_data_pipeline.main \\"
    echo "      --num-scenes 2000 \\"
    echo "      --output-dir ./playground/data/shapellm/scaffold_v3"
    exit 1
fi

# Run training
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LLM_VERSION \
    --version plain \
    --data_path $DATA_PATH \
    --point_folder $PCS_PATH \
    --vision_tower ReConV2/cfgs/pretrain/large/openshape.yaml \
    --vision_tower_path $VISION_TOWER_PATH \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
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
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --logging_dir $LOG_DIR

echo "=============================================================="
echo "Stage 1 V3 Training Complete!"
echo "=============================================================="
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Next step: Run Stage 2 V3 training with:"
echo "  bash scripts/finetune_scaffold_stage2_v3.sh"
echo "=============================================================="
