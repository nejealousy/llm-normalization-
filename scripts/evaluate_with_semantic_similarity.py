import json
import csv
import os
import re
from sentence_transformers import SentenceTransformer, util

# === Конфигурация ===
TEST_DATASET_PATH = "data/test_dataset.csv"
LLM_OUTPUTS_PATH = "data/llm_outputs.json"
SIMILARITY_THRESHOLD = 0.75
MODEL_NAME = 'sentence-transformers/paraphrase-mpnet-base-v2'

# === Модель эмбеддингов ===
model = SentenceTransformer(MODEL_NAME)

def normalize(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())

def factify(data_dict):
    return [f"{normalize(k)}: {normalize(v)}" for k, v in data_dict.items() if v and normalize(v) not in ("none", "неизвестно")]

def load_data():
    with open(TEST_DATASET_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_data = [(row["raw_text"], json.loads(row["expected_attrs"])) for row in reader]

    with open(LLM_OUTPUTS_PATH, encoding="utf-8") as f:
        llm_outputs = json.load(f)

    return test_data, llm_outputs

def compute_factual_overlap(expected_facts, predicted_facts):
    if not predicted_facts:
        return 0, 0, 0

    expected_embeds = model.encode(expected_facts, convert_to_tensor=True)
    predicted_embeds = model.encode(predicted_facts, convert_to_tensor=True)

    tp = 0
    matched_preds = set()

    for i, ef in enumerate(expected_facts):
        scores = util.cos_sim(expected_embeds[i], predicted_embeds)[0]
        best_score = scores.max().item()
        best_idx = scores.argmax().item()

        if best_score >= SIMILARITY_THRESHOLD:
            tp += 1
            matched_preds.add(best_idx)

    fp = len(predicted_facts) - len(matched_preds)
    fn = len(expected_facts) - tp

    return tp, fp, fn

def evaluate_facts():
    test_data, llm_outputs = load_data()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for raw_text, expected in test_data:
        predicted = llm_outputs.get(raw_text, {})

        exp_facts = factify(expected)
        pred_facts = factify(predicted)

        tp, fp, fn = compute_factual_overlap(exp_facts, pred_facts)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print("\n📊 Метрики (по смысловым фактам):")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

if __name__ == "__main__":
    evaluate_facts()
