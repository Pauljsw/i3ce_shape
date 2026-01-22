#!/bin/bash
#
# ReCon++ Fine-tuning on Scaffold Data
# This adapts the point cloud encoder to understand scaffold structures
#

# Configuration
CKPTS="./checkpoints/recon_base.pth"  # Pre-trained ReCon++ checkpoint
CONFIG="ReConV2/cfgs/transfer/base/finetune_scaffold.yaml"
EXP_NAME="recon_scaffold_finetune"
OUTPUT_DIR="./checkpoints/recon_scaffold"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Number of GPUs
NUM_GPUS=4

echo "=========================================="
echo "🔧 ReCon++ Scaffold Fine-tuning"
echo "=========================================="
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CKPTS}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "=========================================="

# Check if pretrained checkpoint exists
if [ ! -f "${CKPTS}" ]; then
    echo "⚠️ Warning: Pretrained checkpoint not found at ${CKPTS}"
    echo "Please download the ReCon++ base checkpoint first."
    echo "You can download from: https://huggingface.co/OpenRobotLab/ReCon_Plus_Plus"
    echo ""
    echo "Or use the ShapeLLM checkpoint:"
    echo "  cp ./checkpoints/shapellm-13b-pt/vision_encoder.pt ${CKPTS}"
    exit 1
fi

# Run fine-tuning
cd /home/user/i3ce_shape

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    ReConV2/main.py \
    --config ${CONFIG} \
    --exp_name ${EXP_NAME} \
    --ckpts ${CKPTS} \
    --val_freq 5 \
    --finetune_model \
    2>&1 | tee ${OUTPUT_DIR}/train.log

echo "=========================================="
echo "✅ Fine-tuning completed!"
echo "Checkpoint saved to: ${OUTPUT_DIR}"
echo "=========================================="
