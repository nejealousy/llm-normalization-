import requests
import json
import csv
import os
from time import time

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
    print(f"\n=== RAW RESPONSE for input:\n{text}\n---\n{raw_response}\n===")
    return raw_response

def main():
    input_file = "data/test_dataset.csv"
    output_file = "data/llm_outputs.json"
    os.makedirs("data", exist_ok=True)
    outputs = {}

    with open(input_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            raw_text = row["raw_text"]
            print(f"=== {idx}. Обрабатываем: {raw_text}")
            start = time()
            try:
                response = query_ollama(raw_text)
                elapsed = time() - start
                print(f"⏱ Время запроса: {elapsed:.2f} сек")

                parsed = json.loads(response)
                outputs[raw_text] = parsed
                print("✅ Успешно.")
            except Exception as e:
                elapsed = time() - start
                print(f"❌ Ошибка: {e}")
                print(f"Время запроса (с ошибкой): {elapsed:.2f} сек")
                outputs[raw_text] = {}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Результаты сохранены в '{output_file}'.")

if __name__ == "__main__":
    main()
