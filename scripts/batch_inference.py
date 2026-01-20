#!/usr/bin/env python3
"""Validation/Test set 전체 추론 스크립트"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import load_pts, process_pts
from llava.conversation import conv_templates
from llava.constants import POINT_TOKEN_INDEX


class DataArgs:
    """데이터 전처리 설정"""
    def __init__(self, sample_points_num=10000, with_color=True, occlusion=False):
        self.sample_points_num = sample_points_num
        self.with_color = with_color
        self.occlusion = occlusion


def load_model(model_path, model_base=None):
    """모델 로드"""
    print(f"🔄 Loading model from {model_path}...")

    # LoRA checkpoint 자동 감지
    model_path_obj = Path(model_path)
    is_lora = (model_path_obj / 'adapter_model.bin').exists()

    if is_lora:
        print(f"   Detected LoRA checkpoint")
        model_name = "lora-shapellm"
    else:
        model_name = "shapellm"

    # load_pretrained_model now defaults to bfloat16 to match training
    tokenizer, model, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        torch_dtype=torch.bfloat16  # Match training dtype
    )

    print(f"✅ Model loaded successfully with bfloat16 dtype")
    return tokenizer, model, context_len


def prepare_point_cloud(point_path, data_args):
    """Point cloud 전처리"""
    # Load
    points = load_pts(str(point_path))

    # Process (sampling + normalization)
    points = process_pts(points, data_args)

    return points


def inference_single(model, tokenizer, point_cloud, question, conv_mode="v1"):
    """단일 질문 추론"""
    # Conversation template
    conv = conv_templates[conv_mode].copy()

    # Add question
    question_full = f"<point>\n{question}"
    conv.append_message(conv.roles[0], question_full)
    conv.append_message(conv.roles[1], None)

    # Get prompt
    prompt = conv.get_prompt()

    # Tokenize
    input_ids = tokenizer(prompt).input_ids
    input_ids = torch.as_tensor(input_ids).cuda().unsqueeze(0)

    # Prepare point cloud
    points = point_cloud.cuda().unsqueeze(0)

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            points=points,
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
            use_cache=True,
        )

    # Decode
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Remove prompt
    if output.startswith(prompt):
        output = output[len(prompt):].strip()

    return output


def batch_inference(model, tokenizer, annotations, pcs_dir, data_args, max_samples=None):
    """배치 추론"""
    predictions = []

    # Limit samples if specified
    if max_samples:
        annotations = annotations[:max_samples]

    print(f"\n🚀 Starting inference on {len(annotations)} samples...")

    for ann in tqdm(annotations, desc="Inference"):
        try:
            # Point cloud 파일 경로
            point_file = pcs_dir / ann['point']

            if not point_file.exists():
                print(f"⚠️  File not found: {point_file}")
                continue

            # Point cloud 로드
            point_cloud = prepare_point_cloud(point_file, data_args)

            # 질문 추출
            question = ann['conversations'][0]['value']
            # Remove <point>\n prefix if exists
            if question.startswith('<point>\n'):
                question = question[8:]

            # 추론
            response = inference_single(model, tokenizer, point_cloud, question)

            # 결과 저장 (원본 annotation + model response)
            pred = ann.copy()
            pred['conversations'][1]['value'] = response  # Replace GT with prediction
            predictions.append(pred)

        except Exception as e:
            print(f"❌ Error processing {ann.get('id', 'unknown')}: {e}")
            continue

    print(f"✅ Completed {len(predictions)} predictions")
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Validation/Test set 전체 추론')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Fine-tuned model checkpoint path')
    parser.add_argument('--model_base', type=str, default='qizekun/ShapeLLM_7B_gapartnet_v1.0',
                        help='Base model path')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Dataset directory (contains split.json, pcs/, etc.)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Which split to evaluate (default: val)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output predictions JSON path')
    parser.add_argument('--sample_points_num', type=int, default=10000,
                        help='Number of points to sample')
    parser.add_argument('--with_color', action='store_true',
                        help='Include color information')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (for quick testing)')
    args = parser.parse_args()

    # Paths
    data_dir = Path(args.data_dir)
    pcs_dir = data_dir / 'pcs'
    annotations_file = data_dir / f'instructions_{args.split}.json'

    if not annotations_file.exists():
        print(f"❌ Annotations file not found: {annotations_file}")
        sys.exit(1)

    # Load annotations
    print(f"📂 Loading annotations from {annotations_file}")
    annotations = json.load(open(annotations_file))
    print(f"   Total annotations: {len(annotations)}")

    # Data args
    data_args = DataArgs(
        sample_points_num=args.sample_points_num,
        with_color=args.with_color,
        occlusion=False
    )

    # Load model
    tokenizer, model, context_len = load_model(
        args.model_path,
        args.model_base
    )

    # Batch inference
    predictions = batch_inference(
        model, tokenizer, annotations, pcs_dir, data_args,
        max_samples=args.max_samples
    )

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Predictions saved to {output_path}")
    print(f"   Total predictions: {len(predictions)}")


if __name__ == "__main__":
    main()
