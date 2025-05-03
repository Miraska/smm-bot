import os
import asyncio
import openai
import html
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Union
from datetime import date

from bs4 import BeautifulSoup as bs
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.utils.media_group import MediaGroupBuilder
from aiogram.utils.keyboard import InlineKeyboardBuilder, InlineKeyboardButton
from aiogram.enums.parse_mode import ParseMode
from aiogram.enums.content_type import ContentType
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

import aiohttp
from aiohttp_socks import ProxyConnector

from openai.error import (
    PermissionError as OpenAIPermissionError,
    RateLimitError,
    APIConnectionError,
    OpenAIError
)

load_dotenv(override=True)

# Настройка прокси
PROXY_HOST = os.getenv("PROXY_HOST")
PROXY_PORT = os.getenv("PROXY_PORT")
PROXY_USERNAME = os.getenv("PROXY_USERNAME")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")
proxy_url = f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"

# Константы
NL = '\n'
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHANNEL_ID = os.getenv("CHANNEL_ID")
CHANNEL_URL = os.getenv("CHANNEL_URL")
ALLOWED_USERS = [416064234, 1498695786, 6799175057, 949078033]

GPT_PROMPT = (
    'Сделай минимальные изменения в словах поста, сохраняя HTML-разметку (теги <b>, <i>, <a> и т.д.), также добавь свою разметку (например сделать текст в заголовках жирным с помощью тега <b>), и удали все упоминания аккаунтов и ссылок'
    'типа t.me/ в конце. Удаляй слова "не баг а фича". Размер не должен превышать 1000 символов. Сохраняй исходное форматирование и структуру текста.'
)
CONDENSE_PROMPT = "Сократите этот текст, сохранив основную мысль, HTML-разметку и структуру."
GPT_MAX_TOKENS = 500
GPT_TEMPERATURE = 0.7
LINK_CAPTION = 'INCUBE.AI | ПОДПИСАТЬСЯ'
LINK_APPEND = f'{NL * 2}<a href="{CHANNEL_URL}">{LINK_CAPTION}</a>'
MAX_DAILY_REPOSTS = 555
MAX_SYMBOLS_CAPTION = 1024
MAX_SYMBOLS_MESSAGE = 1000

openai.api_key = OPENAI_API_KEY

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# Инициализация бота и FSM
session_for_bot = AiohttpSession(proxy=proxy_url)
bot = Bot(
    token=TG_BOT_TOKEN,
    session=session_for_bot,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Определение состояний
class BotStates(StatesGroup):
    default = State()
    waiting_for_custom_prompt = State()

# Миддлварь для альбомов
class AlbumMiddleware(BaseMiddleware):
    def __init__(self, latency: float = 0.01):
        self.latency = latency
        self.album_data = {}
        super().__init__()

    async def __call__(
        self,
        handler: Callable[[types.TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: types.Message,
        data: Dict[str, Any]
    ) -> Any:
        id_ = event.media_group_id
        if not id_:
            return await handler(event, data)

        if len(self.album_data.get(id_, [])):
            self.album_data[id_].append(event)
            return None

        self.album_data[id_] = [event]
        await asyncio.sleep(self.latency)

        data['album'] = self.album_data[event.media_group_id]
        return await handler(event, data)

dp.message.middleware(AlbumMiddleware())

# Вспомогательные функции
def format_text(text: str) -> str:
    """Улучшает форматирование текста, сохраняя переносы строк и убирая лишние пробелы внутри строк."""
    lines = text.split('\n')
    formatted_lines = [' '.join(line.split()) for line in lines]
    return '\n'.join(formatted_lines).strip()

def clean_html_for_telegram(text: str) -> str:
    """Очищает HTML и форматирует текст для Telegram, сохраняя теги и переносы строк."""
    if not text:
        return ""
    text = text.replace("\ufeff", "").replace("\u200b", "")
    soup = bs(text, 'html.parser')
    allowed_tags = {"b", "strong", "i", "em", "u", "ins", "s", "strike", "del", "code", "pre", "a", "span"}
    for tag in soup.find_all():
        if tag.name not in allowed_tags:
            tag.unwrap()
        else:
            allowed_attrs = ["href"] if tag.name == "a" else []
            for attr in list(tag.attrs):
                if attr not in allowed_attrs:
                    del tag[attr]
    cleaned_text = str(soup)
    cleaned_text = format_text(cleaned_text)
    return html.unescape(cleaned_text)

def truncate_html(text: str, trunc: int) -> str:
    """Обрезает текст до указанной длины, сохраняя HTML-структуру."""
    soup = bs(text, 'html.parser')
    if len(soup.get_text()) <= trunc:
        return str(soup)
    truncated_raw = soup.get_text()[:trunc]
    truncated_soup = bs(truncated_raw, 'html.parser')
    return str(truncated_soup)

async def chat_completion(prompt: str, custom_prompt: Optional[str] = None) -> Dict[str, Union[bool, str]]:
    if not prompt:
        return {"ok": True, "text": "", "error": ""}
    system_message = custom_prompt if custom_prompt else GPT_PROMPT
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=GPT_MAX_TOKENS,
            temperature=GPT_TEMPERATURE
        )
        msg = response.choices[0].message["content"]
        return {"ok": True, "text": msg.strip() if msg else "", "error": ""}
    except Exception as e:
        logging.error(f"Error in chat_completion: {e}")
        return {"ok": False, "text": "", "error": str(e)}

async def rewrite(text_to_send: str, trunc: Optional[int] = MAX_SYMBOLS_CAPTION, custom_prompt: Optional[str] = None) -> Dict[str, Union[bool, str]]:
    if not text_to_send:
        return {"ok": True, "text": "", "error": ""}
    
    # Формируем полный промпт с учетом custom_prompt
    full_prompt = custom_prompt if custom_prompt else GPT_PROMPT

    print('text_to_send:', text_to_send)
    print('trunc:', trunc)
    print('full_prompt:', full_prompt)
    
    # Первый этап: модификация текста
    result1 = await chat_completion(text_to_send, custom_prompt=full_prompt)
    if not result1["ok"]:
        return result1
    
    modified_text = result1["text"].strip()
    if not modified_text:
        return {"ok": False, "text": "", "error": "OpenAI вернул пустой ответ на первом этапе."}
    
    # Второй этап: сокращение текста
    result2 = await chat_completion(modified_text, custom_prompt=CONDENSE_PROMPT)
    if not result2["ok"]:
        return result2
    
    condensed_text = result2["text"].strip()
    if not condensed_text:
        return {"ok": False, "text": "", "error": "OpenAI вернул пустой ответ на втором этапе."}
    
    # Добавляем ссылку и обрезаем при необходимости
    if trunc:
        max_content_length = trunc - len(LINK_APPEND)
        truncated = truncate_html(condensed_text, max_content_length)
        final_text = f'{truncated}{LINK_APPEND}'
    else:
        final_text = f'{condensed_text}{LINK_APPEND}'
    
    # Очищаем HTML для Telegram
    final_text = clean_html_for_telegram(final_text)
    
    return {"ok": True, "text": final_text, "error": ""}

# Клавиатуры
def get_preview_keyboard(allow_edit=True):
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="Отправить", callback_data="confirm"))
    if allow_edit:
        builder.add(InlineKeyboardButton(text="Редактировать", callback_data="edit"))
    builder.add(InlineKeyboardButton(text="Отменить", callback_data="cancel"))
    return builder.as_markup()

def get_edit_keyboard():
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="Редактировать текущий текст", callback_data="edit_current_text"))
    builder.add(InlineKeyboardButton(text="Отмена", callback_data="cancel_edit"))
    return builder.as_markup()

preview_keyboard = get_preview_keyboard(allow_edit=True)
album_preview_keyboard = get_preview_keyboard(allow_edit=False)
edit_keyboard = get_edit_keyboard()

# Функция отправки превью (исправленная)
async def send_preview(message: types.Message, state: FSMContext):
    data = await state.get_data()
    preview_data = data.get("preview_data")
    if not preview_data:
        await message.edit_text("❌ Ошибка: данные для превью отсутствуют.")
        return
    message_type = preview_data['type']
    if message_type == 'text':
        # Отправляем новое сообщение вместо редактирования исходного
        preview_msg = await bot.send_message(
            message.chat.id,
            preview_data['text'],
            parse_mode=ParseMode.HTML,
            reply_markup=preview_keyboard
        )
    elif message_type in ['photo', 'video', 'animation', 'document']:
        if message_type == 'photo':
            preview_msg = await bot.send_photo(
                message.chat.id, preview_data['media'], caption=preview_data.get('caption'),
                parse_mode=ParseMode.HTML, reply_markup=preview_keyboard
            )
        elif message_type == 'video':
            preview_msg = await bot.send_video(
                message.chat.id, preview_data['media'], caption=preview_data.get('caption'),
                parse_mode=ParseMode.HTML, reply_markup=preview_keyboard
            )
        elif message_type == 'animation':
            preview_msg = await bot.send_animation(
                message.chat.id, preview_data['media'], caption=preview_data.get('caption'),
                parse_mode=ParseMode.HTML, reply_markup=preview_keyboard
            )
        elif message_type == 'document':
            preview_msg = await bot.send_document(
                message.chat.id, preview_data['media'], caption=preview_data.get('caption'),
                parse_mode=ParseMode.HTML, reply_markup=preview_keyboard
            )
    elif message_type == 'media_group':
        media_group = []
        for item in preview_data['media']:
            if item['type'] == 'photo':
                media_group.append(types.InputMediaPhoto(media=item['media'], caption=item.get('caption'), parse_mode=item.get('parse_mode')))
            elif item['type'] == 'video':
                media_group.append(types.InputMediaVideo(media=item['media'], caption=item.get('caption'), parse_mode=item.get('parse_mode')))
        await bot.send_media_group(message.chat.id, media_group)
        preview_msg = await message.edit_text("Превью альбома выше", reply_markup=album_preview_keyboard)
    await state.update_data(preview_message_id=preview_msg.message_id)
    await message.delete()

# Хендлеры
@dp.message(CommandStart())
async def start_handler(message: types.Message, state: FSMContext):
    await message.answer(f'Приветствую, {message.from_user.full_name}!')
    await state.set_state(BotStates.default)

@dp.message(BotStates.default, ~F.media_group_id)
async def message_handler(message: types.Message, state: FSMContext):
    if message.from_user.id not in ALLOWED_USERS:
        logging.info(f'Отказано в доступе пользователю ID {message.from_user.id}')
        return

    is_text = (message.content_type == ContentType.TEXT)
    text_to_send = message.html_text or message.caption
    if not text_to_send:
        return

    data = (await state.get_data()) or {}
    today = date.today().isoformat()
    cnt = data.get(today, 0)
    if cnt >= MAX_DAILY_REPOSTS:
        await message.answer(f'🛑 Превышен суточный лимит в {MAX_DAILY_REPOSTS} сообщений.')
        return

    await message.answer("⏳ Обрабатываю ваше сообщение...")
    rewrite_result = await rewrite(text_to_send=text_to_send, trunc=MAX_SYMBOLS_MESSAGE if is_text else MAX_SYMBOLS_CAPTION)
    if not rewrite_result["ok"]:
        await message.answer(f"❌ Ошибка: {rewrite_result['error']}")
        return
    final_text = rewrite_result["text"].strip()
    if not final_text:
        await message.answer("❌ OpenAI вернул пустой ответ.")
        return

    await message.answer("✅ Текст сгенерирован, отправляю превью...")
    preview_data = {
        'type': 'text' if is_text else message.content_type,
        'text': final_text if is_text else None,
        'caption': final_text if not is_text else None,
        'media': (message.photo[-1].file_id if message.photo else
                  message.video.file_id if message.video else
                  message.animation.file_id if message.animation else
                  message.document.file_id if message.document else None)
    }
    await state.update_data(preview_data=preview_data, original_text=text_to_send)
    await send_preview(message, state)
    cnt += 1
    await state.update_data({today: cnt})

@dp.message(F.media_group_id)
async def album_handler(message: types.Message, album: list[types.Message], state: FSMContext):
    if message.from_user.id not in ALLOWED_USERS:
        logging.info(f'Отказано в доступе пользователю ID {message.from_user.id}')
        return

    data = (await state.get_data()) or {}
    today = date.today().isoformat()
    cnt = data.get(today, 0)
    if cnt >= MAX_DAILY_REPOSTS:
        await message.answer(f'🛑 Превышен суточный лимит в {MAX_DAILY_REPOSTS} сообщений.')
        return

    await message.answer("⏳ Обрабатываю ваш альбом...")
    media_list = []
    for obj in album:
        cap = obj.html_text or obj.caption
        if cap:
            rewrite_result = await rewrite(cap, MAX_SYMBOLS_CAPTION)
            if not rewrite_result["ok"]:
                await message.answer(f"❌ Ошибка: {rewrite_result['error']}")
                return
            final_cap = rewrite_result["text"].strip()
        else:
            final_cap = LINK_APPEND.strip()
        media = getattr(obj, obj.content_type, None)
        if not media:
            continue
        if obj.content_type == ContentType.PHOTO:
            media = media[-1]
        file_id = getattr(media, 'file_id', None)
        if not file_id:
            continue
        media_list.append({'type': obj.content_type, 'media': file_id, 'caption': final_cap, 'parse_mode': ParseMode.HTML})

    await message.answer("✅ Альбом обработан, отправляю превью...")
    await state.update_data(preview_data={'type': 'media_group', 'media': media_list})
    await send_preview(message, state)
    cnt += 1
    await state.update_data({today: cnt})

# Обработчики кнопок
@dp.callback_query(F.data == "confirm")
async def confirm_handler(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    preview_data = data.get("preview_data")
    if not preview_data:
        await callback.message.edit_text("❌ Ошибка: данные для отправки отсутствуют.")
        return
    chat_id = CHANNEL_ID
    message_type = preview_data['type']
    if message_type == 'text':
        await bot.send_message(chat_id, preview_data['text'], parse_mode=ParseMode.HTML)
    elif message_type in ['photo', 'video', 'animation', 'document']:
        if message_type == 'photo':
            await bot.send_photo(chat_id, preview_data['media'], caption=preview_data.get('caption'), parse_mode=ParseMode.HTML)
        elif message_type == 'video':
            await bot.send_video(chat_id, preview_data['media'], caption=preview_data.get('caption'), parse_mode=ParseMode.HTML)
        elif message_type == 'animation':
            await bot.send_animation(chat_id, preview_data['media'], caption=preview_data.get('caption'), parse_mode=ParseMode.HTML)
        elif message_type == 'document':
            await bot.send_document(chat_id, preview_data['media'], caption=preview_data.get('caption'), parse_mode=ParseMode.HTML)
    elif message_type == 'media_group':
        media_group = []
        for item in preview_data['media']:
            if item['type'] == 'photo':
                media_group.append(types.InputMediaPhoto(media=item['media'], caption=item.get('caption'), parse_mode=item.get('parse_mode')))
            elif item['type'] == 'video':
                media_group.append(types.InputMediaVideo(media=item['media'], caption=item.get('caption'), parse_mode=item.get('parse_mode')))
        await bot.send_media_group(chat_id, media_group)
    if callback.message.photo or callback.message.video:
        await callback.message.edit_caption(caption="✅ Сообщение отправлено в канал.", reply_markup=None)
    else:
        await callback.message.edit_text("✅ Сообщение отправлено в канал.", reply_markup=None)
    await state.clear()

@dp.callback_query(F.data == "edit")
async def edit_handler(callback: types.CallbackQuery, state: FSMContext):
    if callback.message.photo or callback.message.video:
        await callback.message.edit_caption(caption="Выберите способ редактирования:", reply_markup=edit_keyboard)
    else:
        await callback.message.edit_text("Выберите способ редактирования:", reply_markup=edit_keyboard)

@dp.callback_query(F.data == "edit_current_text")
async def edit_current_text_handler(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    preview_data = data.get("preview_data")
    if not preview_data:
        await callback.message.edit_text("❌ Ошибка: данные для редактирования отсутствуют.")
        return

    if callback.message.photo or callback.message.video:
        await callback.message.edit_caption(caption="Отправьте новый промпт и новые медиа (если хотите):", reply_markup=None)
    else:
        await callback.message.edit_text("Отправьте новый промпт и новые медиа (если хотите):", reply_markup=None)
    await state.set_state(BotStates.waiting_for_custom_prompt)

@dp.message(BotStates.waiting_for_custom_prompt)
async def process_custom_prompt(message: types.Message, state: FSMContext):
    custom_prompt = message.text
    if not custom_prompt:
        await message.answer("❌ Промпт не может быть пустым.")
        return
    
    data = await state.get_data()
    original_text = data.get("original_text")
    
    if not original_text:
        await message.answer("❌ Оригинальный текст отсутствует.")
        await state.set_state(BotStates.default)
        return
    
    status_message = await message.answer("⏳ Обрабатываю ваш текст с новым промптом...")
    
    preview_data = data.get("preview_data")
    is_text = preview_data['type'] == 'text'
    max_length = MAX_SYMBOLS_MESSAGE if is_text else MAX_SYMBOLS_CAPTION
    
    # Проверяем, есть ли уже обработанный текст для использования вместо оригинального
    text_to_process = original_text
    if preview_data['type'] == 'text' and preview_data.get('text'):
        # Удаляем ссылку LINK_APPEND, если она есть в конце текста
        current_text = preview_data.get('text')
        if current_text.endswith(LINK_APPEND):
            current_text = current_text[:-len(LINK_APPEND)]
        text_to_process = current_text
    elif preview_data.get('caption'):
        # Удаляем ссылку LINK_APPEND, если она есть в конце подписи
        current_caption = preview_data.get('caption')
        if current_caption.endswith(LINK_APPEND):
            current_caption = current_caption[:-len(LINK_APPEND)]
        text_to_process = current_caption
    
    rewrite_result = await rewrite(text_to_send=text_to_process, trunc=max_length, custom_prompt=custom_prompt)
    
    if not rewrite_result["ok"]:
        await message.answer(f"❌ Ошибка: {rewrite_result['error']}")
        return
    
    final_text = rewrite_result["text"].strip()
    if not final_text:
        await message.answer("❌ OpenAI вернул пустой ответ.")
        return
    
    if preview_data['type'] == 'text':
        preview_data['text'] = final_text
    else:
        preview_data['caption'] = final_text
    
    await state.update_data(preview_data=preview_data)
    await state.set_state(BotStates.default)
    
    try:
        await status_message.edit_text("✅ Текст обработан с новым промптом. Отправляю превью...")
    except Exception as e:
        logging.error(f"Ошибка при обновлении статусного сообщения: {e}")
    
    # Отправляем превью с обновленным текстом
    await send_preview(message, state)

@dp.callback_query(F.data == "cancel")
async def cancel_handler(callback: types.CallbackQuery, state: FSMContext):
    if callback.message.photo or callback.message.video:
        await callback.message.edit_caption(caption="❌ Действие отменено.", reply_markup=None)
    else:
        await callback.message.edit_text("❌ Действие отменено.", reply_markup=None)
    await state.clear()

@dp.callback_query(F.data == "cancel_edit")
async def cancel_edit_handler(callback: types.CallbackQuery, state: FSMContext):
    await send_preview(callback.message, state)

# Точка входа
async def main():
    connector = ProxyConnector.from_url(proxy_url)
    session_for_openai = aiohttp.ClientSession(connector=connector)
    openai.aiosession.set(session_for_openai)
    logging.info("Бот запущен. Ожидаю сообщения...")
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        logging.info("Останавливаем бота...")
        await session_for_openai.close()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот выключен.")