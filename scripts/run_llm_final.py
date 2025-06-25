import requests
import json
import csv
import os
from timing_utils import Timer

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

PROMPT_TEMPLATE = """
Ты — интеллектуальный парсер товарных описаний.  
Тебе на вход приходит строка с названием и характеристиками товара.  
Твоя задача — на основе текста сформировать структурированный JSON с максимально подробным набором характеристик товара, которые можно из него извлечь.

Правила:  
1. Обязательно выдели ключевой атрибут — "Вид продукции".  
2. После этого — попробуй определить все возможные параметры, которые относятся к этому виду продукции.  
3. Если параметр отсутствует или не упоминается — укажи "Неизвестно".  
4. Если в описании есть другие характеристики, создай для них отдельные поля.  
5. Отвечай строго в формате JSON без лишнего текста.

Пример:

Вход:  
"Гайка М10, класс прочности 8, стандарт ГОСТ 5915-70"

Выход:  
{{
  "Вид продукции": "Гайка",
  "Тип гайки": "Неизвестно",
  "Исполнение": "Неизвестно",
  "Диаметр резьбы, мм": "М10",
  "Шаг резьбы, мм": "Неизвестно",
  "Класс прочности": "8",
  "Условное обозначение группы материалов": "Неизвестно",
  "Металлы и сплавы": "Неизвестно",
  "Покрытие изделия": "Неизвестно",
  "Толщина покрытия, мкм": "Неизвестно",
  "Стандарт": "ГОСТ 5915-70"
}}

Вход: {input_text}

Выход:
"""

def query_ollama(text):
    prompt = PROMPT_TEMPLATE.format(input_text=text)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    raw_response = data.get("response", "").strip()
    print(f"\n=== RAW RESPONSE for input:\n{text}\n---\n{raw_response}\n===")  # Выводим полный ответ модели
    return raw_response

import csv
import json
import os
from timing_utils import Timer

# ... твоя функция query_ollama и остальной код

def main():
    input_file = "data/test_dataset.csv"
    output_file = "data/llm_outputs.json"
    os.makedirs("data", exist_ok=True)

    outputs = {}

    total_timer = Timer()
    total_timer.start()

    with open(input_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            raw_text = row["raw_text"]
            print(f"=== {idx}. Обрабатываем: {raw_text}")

            req_timer = Timer()
            req_timer.start()
            try:
                response = query_ollama(raw_text)
                req_timer.stop()

                print(f"⏱ Время запроса: {req_timer.elapsed():.2f} сек")

                parsed = json.loads(response)
                outputs[raw_text] = parsed
                print("✅ Успешно.")
            except Exception as e:
                req_timer.stop()
                print(f"❌ Ошибка обработки: {e}")
                print(f"Время запроса (с ошибкой): {req_timer.elapsed():.2f} сек")
                outputs[raw_text] = {}

    total_timer.stop()
    print(f"\nОбщее время обработки: {total_timer.elapsed():.2f} сек")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Результаты сохранены в {output_file}")

if __name__ == "__main__":
    main()

