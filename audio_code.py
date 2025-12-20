# ЗАПУСК
# py audio_code.py -i "viee.mp4"

# ЧТО СКАЧИВАТЬ
# py -m pip install -U openai-whisper pydub nltk scikit-learn numpy
# py -c "import nltk; nltk.download('punkt_tab')"

# Скачай FFmpeg 
# iwr -useb get.scoop.sh | iex
# Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
# iwr -useb get.scoop.sh | iex
# scoop install ffmpeg
# ffmpeg --version ( должно быть, если ошибка переделывай)

# ВСЕ СКАЧИВАТЬ ЧЕРЕЗ ТЕРМИНАЛ с своЮ PS C:\Users\"ТВОЯ ИМЯ КОМПЬЮТЕРА">

# ЕСЛИ СКАЧАЛ РЕПУ, СНАЧАЛО cd MAIN_CODE в терминал в vs code
# потом уже запускаешь файлы и т.п. 


import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import timedelta
import numpy as np

# Зависимости: whisper, pydub, nltk, sklearn
try:
    import whisper
except Exception:
    print("Ошибка: требуется библиотека 'whisper'. Установите: pip install -U openai-whisper")
    raise

try:
    from pydub import AudioSegment, silence
except Exception:
    print("Ошибка: требуется библиотека 'pydub'. Установите: pip install pydub")
    raise

try:
    import nltk
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    print("Ошибка: требуется 'nltk' и 'scikit-learn'. Установите: pip install nltk scikit-learn")
    raise

# Подготовка NLTK
nltk_packages = ["punkt", "stopwords"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except Exception:
        try:
            nltk.download(pkg)
        except Exception:
            pass

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

RUSSIAN_FILLERS = [
    "эээ", "эм", "мм", "как бы", "типа", "в общем", "это самое", "значит", "ну", "короче",
    "вот", "так сказать", "правда", "собственно", "значит вот"
]
ENGLISH_FILLERS = [
    "um", "uh", "you know", "like", "i mean", "so", "well", "actually", "basically"
]
FILLERS = RUSSIAN_FILLERS + ENGLISH_FILLERS


def extract_audio(input_path: str, output_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-vn", output_path
    ]
    print("Запускаю ffmpeg для извлечения аудио...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Ошибка ffmpeg — проверьте установку.")
        raise


# Создаем кэш для моделей, чтобы не грузить их каждый раз
MODELS_CACHE = {}

def transcribe_whisper(model_name: str, audio_path: str):
    # Проверяем, загружена ли уже эта модель
    if model_name not in MODELS_CACHE:
        print(f"--- Загружаю модель {model_name} в память (первый запуск) ---")
        MODELS_CACHE[model_name] = whisper.load_model(model_name)
    else:
        print(f"--- Использую уже загруженную модель {model_name} ---")
    
    model = MODELS_CACHE[model_name]
    print("Транскрибирую аудио...")
    return model.transcribe(audio_path, verbose=False)

def clear_whisper_cache():
    global MODELS_CACHE
    MODELS_CACHE.clear()
    import gc
    import torch
    gc.collect() # Очистка мусора в Python
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Очистка памяти видеокарты (если есть)
    print("Кэш моделей очищен, память свободна.")

def compute_tempo(total_words: int, duration_seconds: float) -> float:
    minutes = duration_seconds / 60.0
    return 0.0 if minutes <= 0 else total_words / minutes


def analyze_pauses(segments):
    gaps = []
    for i in range(len(segments) - 1):
        end = segments[i]["end"]
        start_next = segments[i + 1]["start"]
        gap = max(0.0, start_next - end)
        gaps.append(round(gap, 3))
    if not gaps:
        return {"gaps": [], "mean": 0.0, "median": 0.0, "long_pauses": []}
    arr = np.array(gaps)
    long_pauses = [g for g in gaps if g >= 1.0]
    return {
        "gaps": gaps,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "long_pauses": long_pauses,
    }


def find_fillers(segments):
    fillers_found = []
    total_counts = Counter()
    pattern_map = [(f, re.compile(r"\b" + re.escape(f) + r"\b", flags=re.IGNORECASE)) for f in FILLERS]
    for seg in segments:
        text = seg["text"]
        for f, pat in pattern_map:
            for m in pat.finditer(text):
                approx_time = seg["start"] + (seg["end"] - seg["start"]) * (m.start() / max(1, len(text)))
                fillers_found.append({"word": f, "segment_text": text.strip(), "time": round(approx_time, 2)})
                total_counts[f] += 1
    return {"list": fillers_found, "counts": dict(total_counts)}


def structure_and_keywords(full_text: str, segments, n_keywords=6):
    duration = segments[-1]["end"] if segments else 0.0
    sents = sent_tokenize(full_text)
    intro_cut = duration * 0.15
    outro_cut = duration * 0.85
    parts = {"intro": [], "body": [], "conclusion": []}
    for seg in segments:
        t_mid = (seg["start"] + seg["end"]) / 2.0
        if t_mid <= intro_cut:
            parts["intro"].append(seg["text"])
        elif t_mid >= outro_cut:
            parts["conclusion"].append(seg["text"])
        else:
            parts["body"].append(seg["text"])

    part_texts = [" ".join(parts["intro"]).strip(), " ".join(parts["body"]).strip(), " ".join(parts["conclusion"]).strip()]
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, stop_words=None, ngram_range=(1, 2))
    try:
        tfidf = vectorizer.fit_transform([t if t else "" for t in part_texts])
        feature_names = np.array(vectorizer.get_feature_names_out())
        keywords = {}
        for i, name in enumerate(["intro", "body", "conclusion"]):
            row = tfidf.getrow(i).toarray().flatten()
            if row.sum() <= 0:
                keywords[name] = []
                continue
            top_n = row.argsort()[-n_keywords:][::-1]
            kw = [feature_names[idx] for idx in top_n if row[idx] > 0]
            keywords[name] = kw
    except:
        words = [w.lower() for w in word_tokenize(full_text) if w.isalpha()]
        sw = set(stopwords.words('russian') + stopwords.words('english'))
        filtered = [w for w in words if w not in sw]
        most = [w for w, _ in Counter(filtered).most_common(n_keywords)]
        keywords = {"intro": most, "body": most, "conclusion": most}

    all_keywords = set(sum([v for v in keywords.values()], []))
    thesis_scores = []
    for sent in sents:
        score = sum(1 for kw in all_keywords if re.search(r"\b" + re.escape(kw) + r"\b", sent, flags=re.IGNORECASE))
        thesis_scores.append((score, sent))
    thesis_scores.sort(reverse=True)
    top_theses = [s for _, s in thesis_scores[:5]]

    return {"keywords": keywords, "top_theses": top_theses}


def generate_recommendations(analysis):
    recs = []
    tempo = analysis.get("tempo_wpm", 0)
    if tempo < 100:
        recs.append("Увеличьте темп: попробуйте 120-150 слов/мин.")
    elif tempo > 170:
        recs.append("Темп слишком высокий: делайте осознанные паузы.")
    else:
        recs.append("Темп нормальный.")

    mean_pause = analysis.get("pauses", {}).get("mean", 0)
    long_pauses = analysis.get("pauses", {}).get("long_pauses", [])
    if mean_pause < 0.3:
        recs.append("Паузы слишком короткие — попробуйте 0.5–1 сек.")
    if long_pauses:
        recs.append(f"Длинные паузы >1с: {len(long_pauses)} — проверьте осознанность.")

    total_fillers = sum(analysis.get("fillers", {}).get("counts", {}).values())
    if total_fillers > 5:
        recs.append(f"Много слов-паразитов ({total_fillers}). Заменяйте их паузами.")
    elif total_fillers > 0:
        recs.append(f"Есть паразитные слова ({total_fillers}). Уменьшайте их количество.")
    else:
        recs.append("Паразитов нет — отлично.")

    if analysis.get("structure", {}).get("keywords", {}):
        recs.append("Укрепите структуру — выделяйте главные переходы.")

    return recs


def seconds_to_hhmmss(s):
    return str(timedelta(seconds=int(round(s))))


def build_report(transcription_result):
    full_text = transcription_result.get("text", "").strip()
    segments = transcription_result.get("segments", [])
    segs = [{"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()} for seg in segments]

    duration = segs[-1]["end"] if segs else 0.0
    words = re.findall(r"\w+", full_text)
    total_words = len(words)

    tempo = compute_tempo(total_words, duration)
    pauses = analyze_pauses(segs)
    fillers = find_fillers(segs)
    structure = structure_and_keywords(full_text, segs)

    analysis = {
        "duration_seconds": duration,
        "total_words": total_words,
        "tempo_wpm": round(tempo, 1),
        "pauses": pauses,
        "fillers": fillers,
        "structure": structure,
    }

    report = {
        "transcription_text": full_text,
        "segments": segs,
        "analysis": analysis,
        "recommendations": generate_recommendations(analysis),
    }
    return report


def save_outputs(report: dict, out_prefix: str = "report"):
    json_path = out_prefix + ".json"
    txt_path = out_prefix + ".txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("AI Speech Coach — Отчёт\n\n")
        dur = report["analysis"]["duration_seconds"]
        f.write(f"Длительность: {seconds_to_hhmmss(dur)} ({dur:.1f} s)\n")
        f.write(f"Слов в речи: {report['analysis']['total_words']}\n")
        f.write(f"Темп (слов/мин): {report['analysis']['tempo_wpm']}\n\n")

        f.write("--- Ключевые тезисы:\n")
        for t in report['analysis']['structure']['top_theses']:
            f.write(f"- {t}\n")

        f.write("\n--- Ключевые слова:\n")
        kws = report['analysis']['structure']['keywords']
        for part in ["intro", "body", "conclusion"]:
            f.write(f"{part}: {', '.join(kws.get(part, []))}\n")

        f.write("\n--- Паразитные слова:\n")
        for item in report['analysis']['fillers']['list'][:20]:
            sentences = re.split(r'(?<=[.!?])\s+', item['segment_text'].replace('\n',' '))
            context = next((s for s in sentences if re.search(r"\b"+re.escape(item['word'])+r"\b", s, re.IGNORECASE)), item['segment_text'])
            f.write(f"{seconds_to_hhmmss(item['time'])} — {item['word']} — {context}\n")

        f.write("\n--- Паузы:\n")
        f.write(f"Средняя: {report['analysis']['pauses']['mean']:.2f}s\n")
        f.write(f">1s: {len(report['analysis']['pauses']['long_pauses'])}\n")

        f.write("\n--- Рекомендации:\n")
        for r in report['recommendations']:
            f.write(f"- {r}\n")

    print(f"Отчёт сохранён: {json_path}, {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="AI Speech Coach — анализ видео")
    parser.add_argument("--input", "-i", required=True, help="Видео файл (mp4, mov и т.д.)")
    parser.add_argument("--model", "-m", default="small", help="Модель Whisper")
    parser.add_argument("--out", "-o", default="report", help="Префикс output файлов")
    args = parser.parse_args()

    input_path = args.input
    model_name = args.model
    out_prefix = args.out

    if not os.path.exists(input_path):
        print(f"Файл не найден: {input_path}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        try:
            extract_audio(input_path, audio_path)
        except Exception as e:
            print("Ошибка извлечения аудио:", e)
            sys.exit(1)

        try:
            result = transcribe_whisper(model_name, audio_path)
        except Exception as e:
            print("Ошибка транскрибации:", e)
            sys.exit(1)

        report = build_report(result)
        save_outputs(report, out_prefix)


if __name__ == "__main__":
    main()
