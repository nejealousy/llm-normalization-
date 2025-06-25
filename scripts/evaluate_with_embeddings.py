import json
import csv
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import re

# --- Параметры ---
THRESHOLD = 0.001
SYNONYMS = {
    "вид продукции": ["тип продукта", "товар", "продукция", "тип", "категория"],
    "бренд": ["марка", "производитель", "бренд", "бренд товара", "компания", "изготовитель", "information о производителе", "бренд производителя"],
    "модель": ["название модели", "модель", "артикул", "код модели", "номер модели", "номер артикула", "модель товара", "серия"],
    "тип": ["тип", "тип амортизатора", "тип батареи", "тип устройства", "тип изделия", "вариант"],
    "размер": ["размер", "габарит", "длина", "ширина", "высота", "толщина", "диаметр", "величина", "объем"],
    "цвет": ["цвет", "оттенок", "окраска"],
    "вес": ["вес", "масса", "weight"],
    "материал": ["материал", "состав", "основа", "материал изделия"],
    "покрытие": ["покрытие", "покрытие изделия", "тип покрытия"],
    "класс прочности": ["класс прочности", "прочность", "стандарт прочности"],
    "стандарт": ["стандарт", "сертификат", "норма", "ГОСТ", "ТУ", "ISO"],
    "мощность": ["мощность", "мощность двигателя", "энергия", "вт", "ватт"],
    "емкость": ["емкость", "объем", "вместимость", "capacity"],
    "напряжение": ["напряжение", "вольтаж", "вольт"],
    "скорость": ["скорость", "частота", "обороты", "rpm"],
    "давление": ["давление", "рабочее давление", "максимальное давление", "ру"],
    "тип батареи": ["тип батареи", "химический состав", "тип аккумулятора"],
    "защита": ["защита", "класс защиты", "ip класс", "степень защиты"],
    "модель двигателя": ["модель двигателя", "тип двигателя"],
    "срок службы": ["срок службы", "гарантия", "ресурс"],
    "особенности": ["особенности", "характеристики", "дополнительные характеристики", "описание"],
    "длина": ["длина", "размер по длине"],
    "ширина": ["ширина", "размер по ширине"],
    "высота": ["высота", "размер по высоте"],
    # Добавь свои варианты
}


model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

# --- Утилиты ---
def normalize(text):
    if text is None:
        return "none"
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s\-.,]", "", text)
    return text

def are_similar(a, b):
    if normalize(a) == normalize(b):
        return True
    emb1 = model.encode(normalize(a), convert_to_tensor=True)
    emb2 = model.encode(normalize(b), convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    return similarity >= THRESHOLD

def get_key_synonyms(key):
    key = normalize(key)
    return [key] + SYNONYMS.get(key, [])

def match_keys(pred_keys, ref_key):
    ref_variants = get_key_synonyms(ref_key)
    for rk in ref_variants:
        for pk in pred_keys:
            if normalize(pk) == rk:
                return pk
    return None

# --- Загрузка данных ---
def load_data():
    with open("data/test_dataset.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_texts = []
        references = []
        for row in reader:
            raw_texts.append(row["raw_text"])
            references.append(json.loads(row["expected_attrs"]))

    with open("data/llm_outputs.json", encoding="utf-8") as f:
        preds_raw = json.load(f)

    predictions = [preds_raw.get(text, {}) for text in raw_texts]
    return raw_texts, references, predictions

# --- Оценка ---
def evaluate():
    raw_texts, references, predictions = load_data()

    tp, fp, fn = 0, 0, 0
    top_errors = []
    key_errors = {}

    for i, (ref, pred, text) in enumerate(zip(references, predictions, raw_texts)):
        pred_keys_used = set()
        for ref_key, ref_val in ref.items():
            matched_key = match_keys(pred.keys(), ref_key)
            if matched_key:
                pred_keys_used.add(matched_key)
                pred_val = pred[matched_key]
                if are_similar(ref_val, pred_val):
                    tp += 1
                else:
                    fn += 1
                    top_errors.append((f"Неверное значение: {ref_key}", ref_val, pred_val, text))
                    key_errors[ref_key] = key_errors.get(ref_key, 0) + 1
            else:
                fn += 1
                top_errors.append((f"Ожидается, но не найден: {ref_key}", ref_val, None, text))
                key_errors[ref_key] = key_errors.get(ref_key, 0) + 1

        for pred_key in pred.keys():
            if pred_key not in pred_keys_used:
                fp += 1
                top_errors.append((f"Лишний ключ: {pred_key}", None, pred[pred_key], text))
                key_errors[pred_key] = key_errors.get(pred_key, 0) + 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    print("\n📊 Метрики:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    print("\n❌ Топ ошибок:")
    for err_type, expected, predicted, text in top_errors[:10]:
        print(f"- {err_type} | Ожид.: {expected} | Предсказ.: {predicted} | Текст: {text[:50]}...")

    print("\n📌 Частые ошибки:")
    for key, count in sorted(key_errors.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"- {key}: {count}")

if __name__ == "__main__":
    print("🔄 Загружаем данные...")
    print("⚙️ Оцениваем...")
    evaluate()
