# что скачивать 
# py -m pip install requests



import json
import requests
import os

# -----------------------------
# НАСТРОЙКИ
# -----------------------------


OPENROUTER_API_KEY = "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"
MODEL = "google/gemini-3-flash-preview"

INPUT_FILE = "report.json"
OUTPUT_TEXT = "feedback.txt"
OUTPUT_JSON = "feedback.json"

# -----------------------------
# ЗАГРУЗКА ФАЙЛА
# -----------------------------
def load_report(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# ФОРМИРОВАНИЕ ПРОМПТА ДЛЯ LLM
# -----------------------------

def build_prompt(data):
    import json
    transcription = data.get("transcription_text", "")
    analysis = data.get("analysis", {})
    recs = data.get("recommendations", [])

    prompt = f"""
Ты — профессиональный коуч по ораторскому искусству. Твоя задача — составить отчет для Telegram.

⚠️ ПРАВИЛА ОФОРМЛЕНИЯ:
- Используй HTML-теги: <b>, <i>, <code>, <blockquote>.
- ВАЖНО: Весь текст внутри каждого раздела (после заголовка) ОБЯЗАТЕЛЬНО оборачивай в тег <blockquote>...</blockquote>.
- Для длинных текстов в разделе FULL используй <blockquote expandable>...</blockquote>.
- Используй эмодзи-буллиты: 💎, ✅, ❌, 📌, 🚀, 💡.

Данные для анализа:
Транскрипт: {transcription}
Аналитика: {json.dumps(analysis, ensure_ascii=False)}
Рекомендации: {json.dumps(recs, ensure_ascii=False)}

СТРУКТУРА ОТВЕТА (ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙ ЭТИ РАЗДЕЛИТЕЛИ):

===SHORT===
🚀 <b>Главный инсайт:</b> 
<blockquote>(текст инсайта)</blockquote>

📌 <b>Топ-3 совета:</b>
<blockquote>
• (совет 1)
• (совет 2)
• (совет 3)
</blockquote>

===FULL===
📊 <b>ПОДРОБНЫЙ АНАЛИЗ</b>
<blockquote expandable>
(весь подробный анализ, сильные стороны, зоны роста и план действий пиши здесь внутри одной большой сворачиваемой цитаты)
</blockquote>

===RESOURCES===
🎓 <b>БИБЛИОТЕКА ОРАТОРА</b>

<blockquote>
<b>📺 СМОТРЕТЬ:</b> <a href="https://www.ted.com/playlists/224/how_to_deliver_a_great_talk">TED: Как выступать блестяще</a>
<i>— Плейлист от лучших спикеров мира. Разбор структуры и подачи.</i>

<b>📖 ЧИТАТЬ:</b> <a href="https://www.litres.ru/book/dzhon-stivens/kak-govorit-chtoby-vas-slushali-63640286/">«Как говорить, чтобы слушали»</a>
<i>— Классика о том, как управлять вниманием через голос.</i>

<b>🛠 ПРАКТИКА:</b> <a href="https://10fastfingers.com/typing-test/russian">Тест скорости мышления</a>
<i>— Развивай навык быстрой подборки слов, чтобы забыть о паузах «эээ...».</i>

<b>💎 ИНСАЙТ:</b>
<i>«Твоя задача — не прочитать текст, а заразить идеей». Пересмотри видео и найди момент, где ты сам веришь в то, что говоришь.</i>
</blockquote>
"""
    return prompt


# -----------------------------
# LLM ЗАПРОС ЧЕРЕЗ OPENROUTER
# -----------------------------
def ask_llm(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Speech Feedback Script",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Ты мастер ораторского анализа."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1500
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"OpenRouter error {response.status_code}: {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


# -----------------------------
# СОХРАНЕНИЕ РЕЗУЛЬТАТА
# -----------------------------
def save_results(text_feedback):
    with open(OUTPUT_TEXT, "w", encoding="utf-8") as f:
        f.write(text_feedback)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"feedback": text_feedback}, f, ensure_ascii=False, indent=2)


# -----------------------------
# ОСНОВНОЙ ЗАПУСК
# -----------------------------
def main():
    import os

    def main():
        print("Текущая рабочая директория:", os.getcwd())
        print("Путь к INPUT_FILE (относительный):", INPUT_FILE)
        print("Полный путь к INPUT_FILE:", os.path.abspath(INPUT_FILE))
        print("Файл существует?", os.path.exists(INPUT_FILE))
        
    if not os.path.exists(INPUT_FILE):
        print("⚠️  Файл НЕ найден. Проверьте расположение!")
        return
    print("Чтение входного файла...")
    data = load_report(INPUT_FILE)

    print("Формирование промпта...")
    prompt = build_prompt(data)

    print("Отправка запроса в OpenRouter...")
    feedback = ask_llm(prompt)

    print("Сохранение результатов...")
    save_results(feedback)

    print("\nГотово! Фидбек сохранён в файлах:")
    print(f" - {OUTPUT_TEXT}")
    print(f" - {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
