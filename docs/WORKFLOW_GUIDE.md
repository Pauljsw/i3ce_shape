# ShapeLLM Scaffold Missing Detection - Complete Workflow Guide

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE PIPELINE OVERVIEW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Phase 1] Data Generation                                                  │
│      │                                                                      │
│      ▼                                                                      │
│  [Phase 2] Stage 1 Training (Feature Alignment)                            │
│      │                                                                      │
│      ▼                                                                      │
│  [Phase 3] Stage 2 Training (Instruction Tuning)                           │
│      │                                                                      │
│      ▼                                                                      │
│  [Phase 4] Inference (Generate Predictions)                                │
│      │                                                                      │
│      ▼                                                                      │
│  [Phase 5] Evaluation (Rigorous Academic Assessment)                       │
│      │                                                                      │
│      ▼                                                                      │
│  [Phase 6] Visualization (Paper Figures)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Generation

### When to Run
- **First time setup**
- When changing scaffold configurations
- When increasing dataset size

### Command
```bash
cd /home/user/i3ce_shape

# Quick test (1000 scenes)
python -m tools.scaffold_data_pipeline.main \
    --num-scenes 1000 \
    --output-dir ./playground/data/shapellm/scaffold_v2

# Full dataset (3000+ scenes) with ablation data
python -m tools.scaffold_data_pipeline.main \
    --num-scenes 3000 \
    --output-dir ./playground/data/shapellm/scaffold_v2 \
    --with-ablation
```

### Output Files
```
playground/data/shapellm/scaffold_v2/
├── pcs/                           # Point cloud files (.npy)
│   ├── scaffold_00000.npy
│   └── ...
├── stage1_caption_train.json      # Stage 1 학습용 (캡션)
├── stage1_caption_val.json        # Stage 1 검증용
├── stage2_sft_train.json          # Stage 2 학습용 (QA)
├── stage2_sft_val.json            # Stage 2 검증용
├── test_questions.jsonl           # 테스트 질문
├── test_gt.jsonl                  # 테스트 정답
├── ablation_noise/                # Ablation: 랜덤 노이즈
├── ablation_shuffled/             # Ablation: 점 순서 셔플
└── ablation_cross/                # Ablation: 다른 비계 매칭
```

### Verification
```bash
# Check data counts
wc -l playground/data/shapellm/scaffold_v2/*.json
head -c 500 playground/data/shapellm/scaffold_v2/stage1_caption_train.json
```

---

## Phase 2: Stage 1 Training (Feature Alignment)

### Purpose
- mm_projector 학습 (점군 특징 → LLM 임베딩 정렬)
- LLM backbone은 frozen

### When to Run
- After data generation
- When retraining from scratch

### Command
```bash
# Terminal 1: Training
bash scripts/pretrain_scaffold_stage1.sh

# Terminal 2: Tensorboard (optional, 실시간 모니터링)
tensorboard --logdir ./logs --port 6006
# Browser: http://localhost:6006
```

### Duration
- 5 epochs, ~2100 samples
- Estimated: 30-60 minutes on 4x RTX 3090

### Output Files
```
checkpoints/scaffold-stage1-feature-alignment/
├── checkpoint-100/
├── checkpoint-200/
├── ...
├── mm_projector.bin               # ⭐ Stage 2에서 사용
├── trainer_state.json             # 학습 로그
└── config.json

logs/scaffold-stage1-feature-alignment/
└── events.out.tfevents.*          # Tensorboard 로그
```

### Verification
```bash
# Check checkpoint exists
ls -la checkpoints/scaffold-stage1-feature-alignment/

# Check training loss decreased
cat checkpoints/scaffold-stage1-feature-alignment/trainer_state.json | python -c "
import json, sys
data = json.load(sys.stdin)
logs = [l for l in data['log_history'] if 'loss' in l]
print(f'Initial loss: {logs[0][\"loss\"]:.4f}')
print(f'Final loss: {logs[-1][\"loss\"]:.4f}')
"
```

---

## Phase 3: Stage 2 Training (Instruction Tuning)

### Purpose
- LoRA로 LLM fine-tuning
- 부재 탐지 태스크 학습

### When to Run
- After Stage 1 completion
- Requires mm_projector.bin from Stage 1

### Command
```bash
# Terminal 1: Training
bash scripts/finetune_scaffold_stage2.sh

# Terminal 2: Tensorboard
tensorboard --logdir ./logs --port 6006
```

### Duration
- 3 epochs, ~10000 samples
- Estimated: 2-4 hours on 4x RTX 3090

### Output Files
```
checkpoints/scaffold-stage2-instruction-tuning/
├── checkpoint-100/
├── checkpoint-200/
├── ...
├── adapter_model.bin              # LoRA weights
├── adapter_config.json
├── trainer_state.json
└── config.json

logs/scaffold-stage2-instruction-tuning/
└── events.out.tfevents.*
```

---

## Phase 4: Inference (Generate Predictions)

### Purpose
- 테스트 데이터에 대한 모델 예측 생성

### When to Run
- After Stage 2 completion
- Before evaluation

### Command
```bash
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/scaffold-stage2-instruction-tuning \
    --model-base qizekun/ShapeLLM_7B_gapartnet_v1.0 \
    --question-file ./playground/data/shapellm/scaffold_v2/test_questions.jsonl \
    --point-folder ./playground/data/shapellm/scaffold_v2/pcs \
    --answers-file ./outputs/test_answers.jsonl \
    --temperature 0.2
```

### For Ablation Studies
```bash
# Noise ablation
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/scaffold-stage2-instruction-tuning \
    --model-base qizekun/ShapeLLM_7B_gapartnet_v1.0 \
    --question-file ./playground/data/shapellm/scaffold_v2/ablation_noise/questions.jsonl \
    --point-folder ./playground/data/shapellm/scaffold_v2/ablation_noise/pcs \
    --answers-file ./outputs/ablation_noise_answers.jsonl

# Repeat for other ablations...
```

### Output
```
outputs/
├── test_answers.jsonl             # 메인 예측
├── ablation_noise_answers.jsonl   # Ablation 예측들
├── ablation_shuffled_answers.jsonl
└── ablation_cross_answers.jsonl
```

---

## Phase 5: Evaluation (Rigorous Academic Assessment)

### Purpose
- 종합적인 성능 평가
- Data Leakage 검증 (Ablation)
- 논문용 메트릭 생성

### When to Run
- After inference completion
- Before paper writing

### Command
```bash
# Full evaluation with ablation verification
python tools/evaluate_scaffold_rigorous.py \
    --predictions ./outputs/test_answers.jsonl \
    --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
    --ablation-dir ./playground/data/shapellm/scaffold_v2 \
    --output-dir ./evaluation_results

# Quick evaluation (no ablation)
python tools/evaluate_scaffold_rigorous.py \
    --predictions ./outputs/test_answers.jsonl \
    --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
    --quick \
    --output-dir ./evaluation_results
```

### Output
```
evaluation_results/
├── evaluation_report.json         # 전체 결과 (JSON)
├── results_table.tex              # LaTeX 테이블 (논문용)
└── visualization_data.json        # 그래프용 데이터
```

---

## Phase 5.5: GPT-based Evaluation (Optional but Recommended)

### Purpose
- GPT-4/3.5를 사용한 의미적 유사도 평가
- 5가지 카테고리별 점수 (Binary, Component Type, Count, Location, Overall)
- 논문에서 많이 사용되는 평가 방식

### When to Run
- After Phase 5 (rule-based evaluation)
- When you need semantic evaluation for paper

### Command
```bash
# Detailed evaluation (5 categories)
python tools/gpt_evaluate_scaffold.py \
    --predictions ./outputs/test_answers.jsonl \
    --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
    --questions ./playground/data/shapellm/scaffold_v2/test_questions.jsonl \
    --output-dir ./evaluation_results \
    --openai-key YOUR_OPENAI_API_KEY \
    --model gpt-4

# Simple scoring (faster, cheaper)
python tools/gpt_evaluate_scaffold.py \
    --predictions ./outputs/test_answers.jsonl \
    --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
    --questions ./playground/data/shapellm/scaffold_v2/test_questions.jsonl \
    --output-dir ./evaluation_results \
    --openai-key YOUR_OPENAI_API_KEY \
    --model gpt-3.5-turbo \
    --simple

# Test with limited samples first
python tools/gpt_evaluate_scaffold.py \
    --predictions ./outputs/test_answers.jsonl \
    --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
    --questions ./playground/data/shapellm/scaffold_v2/test_questions.jsonl \
    --output-dir ./evaluation_results \
    --openai-key YOUR_OPENAI_API_KEY \
    --max-samples 50
```

### Output
```
evaluation_results/
├── gpt_eval_results.jsonl         # 개별 샘플 점수
└── gpt_eval_summary.json          # 카테고리별 평균 점수
```

### GPT Evaluation Categories

| Category | Description | Scoring |
|----------|-------------|---------|
| Binary Detection | Yes/No 정답 여부 | 100=정답, 0=오답 |
| Component Type | 부재 종류 식별 | 100=완벽, 50=부분, 0=오류 |
| Count Accuracy | 개수 정확도 | 100=정확, 70=±1, 40=±2 |
| Location Description | 위치 설명 정확도 | 100=정확, 50=부분 |
| Overall Quality | 전체 품질 | 종합 평가 |

### Cost Estimation
- GPT-3.5-turbo: ~$0.001-0.002 per sample
- GPT-4: ~$0.03-0.05 per sample
- 1000 samples with GPT-4 ≈ $30-50

### Metrics Explained

| Category | Metric | Description |
|----------|--------|-------------|
| **Binary** | Accuracy | Yes/No 정답률 |
| | Precision | TP / (TP + FP) |
| | Recall | TP / (TP + FN) |
| | F1 | 2 × P × R / (P + R) |
| **Component** | Type Accuracy | 부재 종류별 정확도 |
| **Count** | Exact Match | 개수 정확히 맞춤 |
| | ±1 Tolerance | 1개 오차 허용 |
| **BBox** | IoU@0.5 | 50% 이상 겹침 비율 |
| | IoU@0.25 | 25% 이상 겹침 비율 |
| | Detection Rate | GT 대비 탐지율 |

---

## Phase 6: Visualization (Paper Figures)

### Purpose
- 논문용 그래프 생성
- Loss curves, comparison charts

### When to Run
- After training completion
- After evaluation

### Command
```bash
# Generate all figures
python tools/paper_visualization.py \
    --all \
    --output-dir ./paper_figures

# Specific plots
python tools/paper_visualization.py --loss-curve
python tools/paper_visualization.py --combined
```

### Output
```
paper_figures/
├── training_loss.png              # Loss 곡선
├── training_loss.pdf              # PDF 버전
├── training_loss.svg              # SVG 버전
├── combined_training.png          # Stage 1+2 비교
├── combined_training.pdf
└── training_summary.json          # 수치 요약
```

---

## Quick Reference: Complete Pipeline Commands

```bash
# ==================== FULL PIPELINE ====================

# 1. Data Generation
python -m tools.scaffold_data_pipeline.main --num-scenes 3000 --with-ablation

# 2. Stage 1 Training
bash scripts/pretrain_scaffold_stage1.sh

# 3. Stage 2 Training
bash scripts/finetune_scaffold_stage2.sh

# 4. Inference
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/scaffold-stage2-instruction-tuning \
    --model-base qizekun/ShapeLLM_7B_gapartnet_v1.0 \
    --question-file ./playground/data/shapellm/scaffold_v2/test_questions.jsonl \
    --point-folder ./playground/data/shapellm/scaffold_v2/pcs \
    --answers-file ./outputs/test_answers.jsonl

# 5. Evaluation
python tools/evaluate_scaffold_rigorous.py \
    --predictions ./outputs/test_answers.jsonl \
    --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
    --output-dir ./evaluation_results

# 6. Visualization
python tools/paper_visualization.py --all --output-dir ./paper_figures
```

---

## Troubleshooting

### Stage 1 checkpoint not found
```bash
# Check if mm_projector.bin exists
ls checkpoints/scaffold-stage1-feature-alignment/mm_projector.bin

# If not, check for checkpoint directories
ls checkpoints/scaffold-stage1-feature-alignment/checkpoint-*/
```

### CUDA Out of Memory
```bash
# Reduce batch size in training scripts
# Stage 1: per_device_train_batch_size: 2 → 1
# Stage 2: per_device_train_batch_size: 4 → 2
```

### Low BBox IoU
- Check bbox format in model outputs
- Increase training epochs
- Review if questions have proper bbox context

### High Binary Accuracy but Low IoU (Data Leakage)
- Run ablation studies to verify
- Check question templates are uniform
- Ensure model isn't using text shortcuts

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Data Generation | 5-30 min | None |
| Stage 1 Training | 30-60 min | Data |
| Stage 2 Training | 2-4 hours | Stage 1 |
| Inference | 30-60 min | Stage 2 |
| Evaluation | 5-10 min | Inference |
| Visualization | 1-2 min | Training logs |

**Total: ~4-6 hours** (for 3000 samples on 4x RTX 3090)
