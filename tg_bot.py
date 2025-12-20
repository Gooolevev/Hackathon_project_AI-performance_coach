import asyncio
import os
import logging
import tempfile
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart, Command
from aiogram.types import BotCommand, BotCommandScopeDefault, InlineKeyboardButton, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv

# Импорт ваших функций (убедитесь, что файлы audio_code.py и LLM.py в той же папке)
from audio_code import extract_audio, transcribe_whisper, build_report
from LLM import build_prompt, ask_llm

# -----------------------------
# Настройка состояний
# -----------------------------
class AnalysisStates(StatesGroup):
    waiting_for_video = State()
    viewing_report = State()

# -----------------------------
# Настройка и Логирование
# -----------------------------
load_dotenv()
# ВАЖНО: Замените 'ВАШ_ТОКЕН' на реальный токен в .env файле или здесь
BOT_TOKEN = os.getenv("BOT_TOKEN", "8598672575:AAEkb0DClX-pkLjjpX2bEGvNpiuAeP4c5Lo")

TEMP_FOLDER = "temp_videos"
os.makedirs(TEMP_FOLDER, exist_ok=True)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Вспомогательные функции
# -----------------------------
def get_report_keyboard():
    """Создает кнопки для выбора раздела отчета"""
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="📌 Кратко", callback_data="show_short"))
    builder.row(InlineKeyboardButton(text="📘 Подробно", callback_data="show_full"))
    builder.row(InlineKeyboardButton(text="🎓 Ресурсы", callback_data="show_resources"))
    return builder.as_markup()

def split_llm_response(text: str):
    parts = {"short": "Нет данных", "full": "Нет данных", "resources": "Нет данных"}
    if "===SHORT===" in text:
        parts["short"] = text.split("===SHORT===")[1].split("===FULL===")[0]
    if "===FULL===" in text:
        parts["full"] = text.split("===FULL===")[1].split("===RESOURCES===")[0]
    if "===RESOURCES===" in text:
        parts["resources"] = text.split("===RESOURCES===")[1]
    return {k: v.strip() for k, v in parts.items()}

def get_video_summary_from_report(report_dict):
    prompt = build_prompt(report_dict)
    feedback = ask_llm(prompt)
    return feedback

# -----------------------------
# Хендлеры команд
# -----------------------------
async def set_commands():
    commands = [
        BotCommand(command="start", description="🚀 Запустить бота"),
        BotCommand(command="help", description="ℹ️ Помощь"),
    ]
    await bot.set_my_commands(commands, BotCommandScopeDefault())

@dp.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "👋 <b>Привет! Я AI-тренер выступлений.</b>\n\n"
        "Отправь мне видео (MP4) со своим выступлением, и я проанализирую твою речь.",
        parse_mode=ParseMode.HTML
    )

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(
        "ℹ️ <b>Как это работает?</b>\n"
        "1. Пришли видеофайлом или кружочком.\n"
        "2. Я извлеку звук и прогоню через нейросеть.\n"
        "3. Ты получишь структурированный фидбек.",
        parse_mode=ParseMode.HTML
    )

# -----------------------------
# Хендлер видео
# -----------------------------
@dp.message(F.video)
async def handle_video(message: types.Message, state: FSMContext):
    status_msg = await message.answer("📥 <b>Скачиваю видео...</b>", parse_mode=ParseMode.HTML)
    
    video = message.video
    file_path = os.path.join(TEMP_FOLDER, f"{video.file_unique_id}.mp4")

    try:
        # Скачивание
        file_info = await bot.get_file(video.file_id)
        await bot.download_file(file_info.file_path, file_path)
        
        await status_msg.edit_text("🧠 <b>Анализирую аудио и текст...</b>", parse_mode=ParseMode.HTML)
        
        # Обработка (в отдельном потоке, чтобы не блокировать бота)
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.wav")
            
            # Синхронные вызовы через run_in_executor или to_thread
            await asyncio.to_thread(extract_audio, file_path, audio_path)
            transcription_result = await asyncio.to_thread(transcribe_whisper, "large-v3", audio_path)
            report = build_report(transcription_result)
            summary = await asyncio.to_thread(get_video_summary_from_report, report)
        
        # Разделяем ответ и сохраняем в FSM
        parts = split_llm_response(summary)
        await state.update_data(report_parts=parts)
        await state.set_state(AnalysisStates.viewing_report)
        
        await status_msg.delete()
        await message.reply(
            "✅ <b>Анализ готов!</b> Выберите раздел для просмотра:",
            reply_markup=get_report_keyboard(),
            parse_mode=ParseMode.HTML
        )

    except Exception as e:
        logging.error(f"Ошибка: {e}")
        await message.reply("❌ Произошла ошибка при обработке видео.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# -----------------------------
# Хендлер кнопок (Callback)
# -----------------------------
@dp.callback_query(AnalysisStates.viewing_report)
async def process_report_selection(callback: CallbackQuery, state: FSMContext):
    user_data = await state.get_data()
    parts = user_data.get("report_parts")
    
    if not parts:
        await callback.answer("Данные устарели, отправьте видео снова.", show_alert=True)
        return

    # Определяем, какую часть текста показать
    if callback.data == "show_short":
        text = f"📌 <b>Краткий гайд:</b>\n\n{parts['short']}"
    elif callback.data == "show_full":
        text = f"📘 <b>Подробный разбор:</b>\n\n{parts['full']}"
    elif callback.data == "show_resources":
        text = f"🎓 <b>Полезные ресурсы:</b>\n\n{parts['resources']}"
    else:
        await callback.answer()
        return

    # Редактируем сообщение, подставляя текст и оставляя кнопки
    try:
        await callback.message.edit_text(
            text,
            reply_markup=get_report_keyboard(),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
    except Exception:
        # Если текст совпадает с текущим, aiogram выбросит ошибку, просто игнорируем
        pass
    
    await callback.answer()

# -----------------------------
# Запуск
# -----------------------------
async def main():
    await set_commands()
    print("Бот запущен и готов к работе...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен")