#!/bin/bash
# ==============================================================================
# Full Training Pipeline V3 for Scaffold Missing Detection
# ==============================================================================
#
# Pipeline Steps:
#   1. Generate V3 data (no text shortcuts, diverse questions)
#   2. (Optional) Fine-tune ReCon++ on scaffold data
#   3. Stage 1: Feature alignment (10 epochs)
#   4. Stage 2: Instruction tuning (5 epochs)
#   5. Evaluation
#
# Environment: 4x RTX 3090 (24GB each)
# Total estimated time: ~24-48 hours
#
# ==============================================================================

set -e

# Configuration
NUM_SCENES=2000
DATA_DIR="./playground/data/shapellm/scaffold_v3"
RECON_FINETUNE=false  # Set to true to fine-tune ReCon++

echo "=============================================================="
echo "🏗️ Scaffold Missing Detection - Full Pipeline V3"
echo "=============================================================="
echo "Scenes: ${NUM_SCENES}"
echo "Data Dir: ${DATA_DIR}"
echo "ReCon++ Fine-tune: ${RECON_FINETUNE}"
echo "=============================================================="

# Step 1: Generate V3 Data
echo ""
echo "📊 Step 1: Generating V3 Data..."
echo "=============================================================="

if [ ! -f "${DATA_DIR}/stage2_sft_train.json" ]; then
    cd /home/user/i3ce_shape
    python tools/generate_scaffold_data_v3.py \
        --num_scenes ${NUM_SCENES} \
        --output_dir ${DATA_DIR} \
        --train_ratio 0.8 \
        --val_ratio 0.1

    # Create test questions file for evaluation
    python -c "
import json
with open('${DATA_DIR}/test_gt.jsonl', 'r') as f:
    lines = f.readlines()

questions = []
for line in lines:
    data = json.loads(line)
    questions.append({
        'question_id': data['question_id'],
        'point': data['point'],
        'text': data.get('answer', '').split('\n')[0] if 'answer' in data else '',
        'task_type': data.get('task_type', '')
    })

with open('${DATA_DIR}/test_questions.jsonl', 'w') as f:
    for q in questions:
        f.write(json.dumps(q) + '\n')

print(f'Created {len(questions)} test questions')
"
    echo "✅ V3 Data generated successfully!"
else
    echo "⏭️ V3 Data already exists, skipping..."
fi

# Step 2: (Optional) Fine-tune ReCon++
if [ "$RECON_FINETUNE" = true ]; then
    echo ""
    echo "🔧 Step 2: Fine-tuning ReCon++ on scaffold data..."
    echo "=============================================================="

    if [ ! -f "./checkpoints/recon_scaffold/ckpt-best.pth" ]; then
        bash scripts/finetune_recon_scaffold.sh
        echo "✅ ReCon++ fine-tuning complete!"
    else
        echo "⏭️ Fine-tuned ReCon++ already exists, skipping..."
    fi
else
    echo ""
    echo "⏭️ Step 2: Skipping ReCon++ fine-tuning (using pretrained)"
fi

# Step 3: Stage 1 Training
echo ""
echo "🎯 Step 3: Stage 1 Feature Alignment (10 epochs)..."
echo "=============================================================="

if [ ! -d "./checkpoints/scaffold-stage1-v3" ]; then
    bash scripts/pretrain_scaffold_stage1_v3.sh
    echo "✅ Stage 1 complete!"
else
    echo "⏭️ Stage 1 checkpoint exists, skipping..."
fi

# Step 4: Stage 2 Training
echo ""
echo "🎯 Step 4: Stage 2 Instruction Tuning (5 epochs)..."
echo "=============================================================="

if [ ! -d "./checkpoints/scaffold-stage2-v3-lora" ]; then
    bash scripts/finetune_scaffold_stage2_v3.sh
    echo "✅ Stage 2 complete!"
else
    echo "⏭️ Stage 2 checkpoint exists, skipping..."
fi

# Step 5: Evaluation
echo ""
echo "📈 Step 5: Running Evaluation..."
echo "=============================================================="

OUTPUT_DIR="./checkpoints/scaffold-stage2-v3-lora"
LLM_VERSION="qizekun/ShapeLLM_7B_gapartnet_v1.0"

# Run inference
python -m llava.eval.model_vqa \
    --model-path ${OUTPUT_DIR} \
    --model-base ${LLM_VERSION} \
    --question-file ${DATA_DIR}/test_questions.jsonl \
    --point-folder ${DATA_DIR}/pcs \
    --answers-file ${OUTPUT_DIR}/test_answers_v3.jsonl \
    --conv-mode scaffold_missing \
    --temperature 0 \
    --num_beams 1

# Run evaluation script
python tools/evaluate_scaffold_rigorous.py \
    --predictions ${OUTPUT_DIR}/test_answers_v3.jsonl \
    --ground-truth ${DATA_DIR}/test_gt.jsonl \
    --output-dir ${OUTPUT_DIR}/evaluation_v3

echo ""
echo "=============================================================="
echo "✅ Full Pipeline Complete!"
echo "=============================================================="
echo ""
echo "📁 Outputs:"
echo "  - Data: ${DATA_DIR}"
echo "  - Stage 1: ./checkpoints/scaffold-stage1-v3"
echo "  - Stage 2: ./checkpoints/scaffold-stage2-v3-lora"
echo "  - Results: ${OUTPUT_DIR}/evaluation_v3"
echo ""
echo "📊 View evaluation results:"
echo "  cat ${OUTPUT_DIR}/evaluation_v3/detailed_report.json"
echo "=============================================================="
