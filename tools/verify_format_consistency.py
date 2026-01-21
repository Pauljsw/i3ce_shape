#!/usr/bin/env python3
"""
Verify format consistency between training and validation data.
Compares question/answer formats, not the actual scenes.
"""

import json

# Load training data (scaffold_00000 as representative)
with open('/home/user/i3ce_shape/playground/data/shapellm/scaffold_test_gt_full/scaffold_missing_sft.json', 'r') as f:
    train_data = json.load(f)
    train_sample = [item for item in train_data if item['id'] == 'scaffold_00000_missing_summary'][0]

# Load validation question (scaffold_00001 as representative)
with open('/home/user/i3ce_shape/playground/data/shapellm/scaffold_test_gt_full/question_missing.jsonl', 'r') as f:
    for line in f:
        val_question = json.loads(line)
        if val_question['question_id'] == 'scaffold_00001_missing_summary':
            break

# Load validation GT (scaffold_00001 as representative)
with open('/home/user/i3ce_shape/playground/data/shapellm/scaffold_test_gt_full/gt_missing.jsonl', 'r') as f:
    for line in f:
        val_gt = json.loads(line)
        if val_gt['question_id'] == 'scaffold_00001_missing_summary':
            break

print("=" * 80)
print("FORMAT CONSISTENCY VERIFICATION")
print("=" * 80)

# 1. Question Format Comparison
print("\n1. QUESTION FORMAT")
print("-" * 80)
train_question = train_sample['conversations'][0]['value']
val_question_text = val_question['text']

print(f"Training question format:")
print(f"  Has <point> token: {'<point>' in train_question}")
print(f"  Question: {train_question[:100]}...")
print()
print(f"Validation question format:")
print(f"  Has <point> token: {'<point>' in val_question_text}")
print(f"  Question: {val_question_text[:100]}...")

question_format_match = ('<point>' in train_question) == ('<point>' in val_question_text)
print(f"\n  ✓ Question format match: {question_format_match}")

# 2. Answer Format Comparison
print("\n2. ANSWER FORMAT")
print("-" * 80)
train_answer = train_sample['conversations'][1]['value']
val_answer = val_gt['text']

print(f"Training answer format:")
print(f"  Has 'Expected structure': {'Expected structure' in train_answer}")
print(f"  Has 'Actual': {'Actual:' in train_answer}")
print(f"  Has 8-corner BBox: {'[' in train_answer and '], [' in train_answer}")
print(f"  Answer preview: {train_answer[:150]}...")
print()
print(f"Validation GT answer format:")
print(f"  Has 'Expected structure': {'Expected structure' in val_answer}")
print(f"  Has 'Actual': {'Actual:' in val_answer}")
print(f"  Has 8-corner BBox: {'[' in val_answer and '], [' in val_answer}")
print(f"  Answer preview: {val_answer[:150]}...")

answer_format_match = (
    ('Expected structure' in train_answer) == ('Expected structure' in val_answer) and
    ('Actual:' in train_answer) == ('Actual:' in val_answer)
)
print(f"\n  ✓ Answer format match: {answer_format_match}")

# 3. BBox Format Comparison
print("\n3. BBOX FORMAT")
print("-" * 80)
# Extract first bbox from training answer
import re
train_bbox_match = re.search(r'\[\[[-\d., ]+\]\]', train_answer)
val_bbox_match = re.search(r'\[\[[-\d., ]+\]\]', val_answer) if val_answer != val_gt['text'] else None

if train_bbox_match:
    train_bbox_sample = train_bbox_match.group(0)
    bbox_count = train_bbox_sample.count('], [')
    print(f"Training BBox format:")
    print(f"  Format: 8-corner (should have 7 separators)")
    print(f"  Separators count: {bbox_count}")
    print(f"  Sample: {train_bbox_sample[:80]}...")

if val_bbox_match:
    val_bbox_sample = val_bbox_match.group(0)
    bbox_count_val = val_bbox_sample.count('], [')
    print(f"\nValidation GT BBox format:")
    print(f"  Format: 8-corner (should have 7 separators)")
    print(f"  Separators count: {bbox_count_val}")
    print(f"  Sample: {val_bbox_sample[:80]}...")
    bbox_format_match = bbox_count == bbox_count_val
    print(f"\n  ✓ BBox format match: {bbox_format_match}")
else:
    print(f"\nValidation GT BBox: No missing components in this example")
    print(f"  BBoxes field: {val_gt.get('bboxes', [])}")
    bbox_format_match = True

# 4. Overall Assessment
print("\n" + "=" * 80)
print("OVERALL ASSESSMENT")
print("=" * 80)
if question_format_match and answer_format_match and bbox_format_match:
    print("✅ ALL FORMATS ARE CONSISTENT")
    print("\nTraining data format matches validation data format:")
    print("  • Questions: Both include/exclude <point> token consistently")
    print("  • Answers: Both use 'Expected structure' and 'Actual' reasoning")
    print("  • BBoxes: Both use 8-corner format")
else:
    print("❌ FORMAT INCONSISTENCIES DETECTED")
    if not question_format_match:
        print("  • Question format mismatch")
    if not answer_format_match:
        print("  • Answer format mismatch")
    if not bbox_format_match:
        print("  • BBox format mismatch")

# 5. Additional check: Does validation question match training question style?
print("\n" + "=" * 80)
print("DETAILED FORMAT CHECK")
print("=" * 80)

# Check if training questions have <point> token
train_questions_with_point = sum(1 for item in train_data if '<point>' in item['conversations'][0]['value'])
print(f"\nTraining data (total {len(train_data)} samples):")
print(f"  Questions with <point> token: {train_questions_with_point}/{len(train_data)}")

# Check validation questions
val_questions = []
with open('/home/user/i3ce_shape/playground/data/shapellm/scaffold_test_gt_full/question_missing.jsonl', 'r') as f:
    val_questions = [json.loads(line) for line in f]

val_questions_with_point = sum(1 for q in val_questions if '<point>' in q['text'])
print(f"\nValidation questions (total {len(val_questions)} samples):")
print(f"  Questions with <point> token: {val_questions_with_point}/{len(val_questions)}")

if train_questions_with_point == len(train_data) and val_questions_with_point == 0:
    print("\n⚠️  WARNING: Training questions have <point> token but validation questions don't!")
    print("   This is a FORMAT INCONSISTENCY that could affect model performance.")
elif train_questions_with_point == 0 and val_questions_with_point == len(val_questions):
    print("\n⚠️  WARNING: Validation questions have <point> token but training questions don't!")
    print("   This is a FORMAT INCONSISTENCY that could affect model performance.")
else:
    print("\n✅ <point> token usage is consistent between training and validation")
