#!/usr/bin/env python3
"""비계 모델 추론 스크립트 (단일 scene 테스트)"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import load_pts, process_pts
from llava.conversation import conv_templates
from llava.constants import POINT_TOKEN_INDEX

def load_model(model_path, model_base=None):
    """모델 로드"""
    print(f"Loading model from {model_path}...")

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
        device_map=None,  # Single GPU (cuda:0)
        device="cuda:0",
        torch_dtype=torch.bfloat16  # Match training dtype
    )

    print(f"✅ Model loaded successfully with bfloat16 dtype")
    return tokenizer, model, context_len

def prepare_point_cloud(point_path, data_args):
    """Point cloud 전처리"""
    # Load
    points = load_pts(point_path)

    # Process (sampling + normalization)
    points = process_pts(points, data_args)

    return points

def inference_single(model, tokenizer, point_cloud, question, conv_mode="v1"):
    """단일 질문 추론"""
    # Conversation template
    conv = conv_templates[conv_mode].copy()

    # Add question
    question = f"<point>\n{question}"
    conv.append_message(conv.roles[0], question)
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

def main():
    parser = argparse.ArgumentParser(description='비계 모델 추론')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Fine-tuned model checkpoint path')
    parser.add_argument('--model_base', type=str, default='qizekun/ShapeLLM_7B_gapartnet_v1.0',
                        help='Base model path')
    parser.add_argument('--point_file', type=str, required=True,
                        help='Point cloud file (.npy)')
    parser.add_argument('--question', type=str, default='Please assess the safety of this scaffold.',
                        help='Question to ask')
    parser.add_argument('--sample_points_num', type=int, default=10000,
                        help='Number of points to sample')
    parser.add_argument('--with_color', action='store_true',
                        help='Include color information')
    args = parser.parse_args()

    # Load model
    tokenizer, model, context_len = load_model(
        args.model_path,
        args.model_base
    )

    # Data args (for preprocessing)
    class DataArgs:
        def __init__(self):
            self.sample_points_num = args.sample_points_num
            self.with_color = args.with_color
            self.occlusion = False

    data_args = DataArgs()

    # Load point cloud
    print(f"\nLoading point cloud: {args.point_file}")
    point_cloud = prepare_point_cloud(args.point_file, data_args)
    print(f"✅ Point cloud shape: {point_cloud.shape}")

    # Inference
    print(f"\nQuestion: {args.question}")
    print("=" * 60)

    response = inference_single(
        model, tokenizer, point_cloud, args.question
    )

    print(f"Response:\n{response}")
    print("=" * 60)

if __name__ == "__main__":
    main()
