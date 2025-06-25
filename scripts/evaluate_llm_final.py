import json
import csv
from difflib import SequenceMatcher
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Словари синонимов ключей и значений ---

KEY_SYNONYMS = {
    "бренд": ["бренд", "марка", "производитель", "информация о производителе"],
    "модель": ["модель", "название модели", "моделль", "марка и модель", "название"],
    "вид продукции": ["вид продукции", "тип продукта", "тип товара", "товар", "вид товара"],
    "тип": ["тип", "тип изделия", "тип фильтра", "тип батареи"],
    # Добавьте другие важные синонимы ключей, если нужно
}

VALUE_SYNONYMS = {
    "да": ["да", "есть", "true", "истина", "встроенный", "yes"],
    "нет": ["нет", "false", "нету", "нету", "отсутствует", "no"],
    "неизвестно": ["неизвестно", "не определено", "нет данных", "unknown", ""],
    # Можно расширять по необходимости
}

IMPORTANT_KEYS = {"вид продукции", "бренд", "модель", "тип"}

# --- Нормализация строк для сравнения ---

def normalize_str(s):
    if s is None:
        return ""
    return str(s).strip().lower()

# Поиск синонима ключа
def normalize_key(key):
    key_norm = normalize_str(key)
    for main_key, syns in KEY_SYNONYMS.items():
        if key_norm == main_key:
            return main_key
        if key_norm in syns:
            return main_key
    return key_norm

# Нормализация значения с учётом синонимов
def normalize_value(val):
    val_norm = normalize_str(val)
    for main_val, syns in VALUE_SYNONYMS.items():
        if val_norm == main_val:
            return main_val
        if val_norm in syns:
            return main_val
    return val_norm

# Частичное сравнение значений (SequenceMatcher)
def values_similar(a, b, threshold=0.7):
    a_norm = normalize_value(a)
    b_norm = normalize_value(b)
    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    return ratio >= threshold

# --- Функция поиска лучшего совпадения ключей между эталоном и предсказанием ---

def match_keys(true_dict, pred_dict, threshold=0.55):
    matches = {}
    used_pred_keys = set()

    for true_key_raw, true_val in true_dict.items():
        true_key = normalize_key(true_key_raw)

        best_pred_key = None
        best_score = 0

        for pred_key_raw in pred_dict:
            if pred_key_raw in used_pred_keys:
                continue
            pred_key = normalize_key(pred_key_raw)
            score = SequenceMatcher(None, true_key, pred_key).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_pred_key = pred_key_raw

        if best_pred_key:
            matches[true_key_raw] = (true_val, pred_dict[best_pred_key])
            used_pred_keys.add(best_pred_key)

    return matches

# --- Главная функция оценки ---

def evaluate(test_dataset_path, llm_output_path):

    # Загрузка тестовых данных
    test_data = []
    with open(test_dataset_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_text = row["raw_text"].strip()
            expected_attrs = json.loads(row["expected_attrs"].replace('""', '"'))
            test_data.append((raw_text, expected_attrs))

    # Загрузка ответов LLM
    with open(llm_output_path, encoding="utf-8") as f:
        llm_outputs = json.load(f)

    # Метрики для всех ключей
    y_true_all = []
    y_pred_all = []

    # Метрики только для важных ключей
    y_true_important = []
    y_pred_important = []

    # Анализ ошибок
    total_true_keys = 0
    total_matched = 0
    total_missing = 0
    total_value_mismatches = 0
    total_extra_keys = 0

    print("\n--- Анализ ошибок ---\n")

    for raw_text, true_attrs in test_data:
        pred_attrs = llm_outputs.get(raw_text, {})

        if not isinstance(pred_attrs, dict):
            pred_attrs = {}

        matched = match_keys(true_attrs, pred_attrs)

        # Подсчет ключей
        true_keys_set = set(normalize_key(k) for k in true_attrs.keys())
        pred_keys_set = set(normalize_key(k) for k in pred_attrs.keys())
        matched_true_keys_set = set(normalize_key(k) for k in matched.keys())

        missing_keys = true_keys_set - matched_true_keys_set
        extra_keys = pred_keys_set - matched_true_keys_set

        # Отчёт по одному товару
        print(f"--- Товар: {raw_text}")

        if missing_keys:
            total_missing += len(missing_keys)
            print(f"❌ Отсутствуют ключи: {missing_keys}")

        if extra_keys:
            total_extra_keys += len(extra_keys)
            print(f"ℹ️ Лишние ключи: {extra_keys}")

        # Проверка значений
        value_mismatches = 0
        for true_key_raw, (true_val, pred_val) in matched.items():
            true_key = normalize_key(true_key_raw)
            total_true_keys += 1

            # Нормализуем значения
            true_val_norm = normalize_value(true_val)
            pred_val_norm = normalize_value(pred_val)

            if values_similar(true_val_norm, pred_val_norm):
                total_matched += 1
                y_true_all.append(1)
                y_pred_all.append(1)
                if true_key in IMPORTANT_KEYS:
                    y_true_important.append(1)
                    y_pred_important.append(1)
            else:
                total_value_mismatches += 1
                value_mismatches += 1
                y_true_all.append(1)
                y_pred_all.append(0)
                if true_key in IMPORTANT_KEYS:
                    y_true_important.append(1)
                    y_pred_important.append(0)
                print(f"⚠️ Несовпадение по значению для '{true_key_raw}': эталон='{true_val}', предсказание='{pred_val}'")

        print(f"✅ Совпавших атрибутов: {len(matched) - value_mismatches}")
        print(f"❌ Несовпадающих по значению: {value_mismatches}")
        print(f"ℹ️ Лишних ключей в предсказании: {len(extra_keys)}")
        print()

    # Финальные метрики
    print("--- Итог ---")
    print(f"Всего атрибутов в эталоне: {total_true_keys}")
    print(f"Совпавших атрибутов: {total_matched}")
    print(f"Отсутствует в предсказании: {total_missing}")
    print(f"Несовпадающих по значению: {total_value_mismatches}")
    print(f"Лишних ключей в предсказании: {total_extra_keys}")

    if y_true_all and y_pred_all:
        precision_all = precision_score(y_true_all, y_pred_all, zero_division=0)
        recall_all = recall_score(y_true_all, y_pred_all, zero_division=0)
        f1_all = f1_score(y_true_all, y_pred_all, zero_division=0)

        precision_imp = precision_score(y_true_important, y_pred_important, zero_division=0) if y_true_important else 0
        recall_imp = recall_score(y_true_important, y_pred_important, zero_division=0) if y_true_important else 0
        f1_imp = f1_score(y_true_important, y_pred_important, zero_division=0) if y_true_important else 0

        print("\n📊 Итоговые метрики качества (все ключи):")
        print(f"Precision: {precision_all:.3f}")
        print(f"Recall:    {recall_all:.3f}")
        print(f"F1 Score:  {f1_all:.3f}")

        print("\n📊 Итоговые метрики качества (важные ключи):")
        print(f"Precision: {precision_imp:.3f}")
        print(f"Recall:    {recall_imp:.3f}")
        print(f"F1 Score:  {f1_imp:.3f}")
    else:
        print("❌ Недостаточно данных для подсчёта метрик.")

if __name__ == "__main__":
    evaluate("data/test_dataset.csv", "data/llm_outputs.json")
