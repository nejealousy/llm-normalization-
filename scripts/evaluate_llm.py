import json
import csv
from difflib import SequenceMatcher
from sklearn.metrics import precision_recall_fscore_support

def normalize_string(s):
    if s is None:
        return "неизвестно"
    return str(s).lower().strip()

def is_similar(a, b, threshold=0.8):
    a_norm = normalize_string(a)
    b_norm = normalize_string(b)
    return SequenceMatcher(None, a_norm, b_norm).ratio() >= threshold

def evaluate(test_data_file, llm_outputs_file, similarity_threshold=0.8):
    # Загрузка тестового датасета
    test_data = []
    with open(test_data_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected_attrs = json.loads(row["expected_attrs"].replace('""', '"'))
            test_data.append((row["raw_text"], expected_attrs))

    # Загрузка ответов модели
    with open(llm_outputs_file, encoding="utf-8") as f:
        llm_outputs = json.load(f)

    y_true = []
    y_pred = []

    for raw_text, true_attrs in test_data:
        pred_attrs = llm_outputs.get(raw_text.strip(), {})

        # Собираем все ключи из эталона и предсказания
        all_keys = set(true_attrs.keys()).union(pred_attrs.keys())

        for key in all_keys:
            true_val = true_attrs.get(key, "неизвестно")
            pred_val = pred_attrs.get(key, "неизвестно")

            y_true.append(normalize_string(true_val))
            y_pred.append(normalize_string(pred_val))

    # Для каждой пары считаем совпадение по порогу похожести
    y_true_binary = []
    y_pred_binary = []

    for true_v, pred_v in zip(y_true, y_pred):
        y_true_binary.append(1)  # В эталоне атрибут есть (или считается "неизвестно")
        y_pred_binary.append(1 if is_similar(true_v, pred_v, similarity_threshold) else 0)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='binary', zero_division=0
    )

    print(f"\n📊 Метрики качества (порог похожести = {similarity_threshold}):")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

if __name__ == "__main__":
    evaluate("data/test_dataset.csv", "data/llm_outputs.json")
