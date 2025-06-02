import os
import asyncio
import openai
import html
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Union
from datetime import date, datetime, timedelta
from enum import Enum

import sqlite3
from contextlib import closing
from typing import List, Tuple

from bs4 import BeautifulSoup as bs
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart, Command
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.utils.media_group import MediaGroupBuilder
from aiogram.utils.keyboard import InlineKeyboardBuilder, InlineKeyboardButton
from aiogram.enums.parse_mode import ParseMode
from aiogram.enums.content_type import ContentType
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.exceptions import TelegramBadRequest

import aiohttp
from aiohttp_socks import ProxyConnector

from openai.error import (
    PermissionError as OpenAIPermissionError,
    RateLimitError,
    APIConnectionError,
    OpenAIError,
)

load_dotenv(override=True)

# Настройка прокси
PROXY_HOST = os.getenv("PROXY_HOST")
PROXY_PORT = os.getenv("PROXY_PORT")
PROXY_USERNAME = os.getenv("PROXY_USERNAME")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")
proxy_url = f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"

# Константы
NL = "\n"
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHANNEL_ID = os.getenv("CHANNEL_ID")
CHANNEL_URL = os.getenv("CHANNEL_URL")
ADMIN_USERNAMES = os.getenv("ADMIN_USERNAMES", "").split(",")
ADMIN_IDS = [416064234, 1498695786, 6799175057, 949078033]

# Лимиты и тарифы
TRIAL_POSTS_LIMIT = 10000
TARIFFS = {
    "basic": {"price": 500, "posts": 100, "days": 30, "name": "Базовый"},
    "standard": {"price": 1000, "posts": 300, "days": 30, "name": "Стандарт"},
    "premium": {"price": 2000, "posts": 1000, "days": 30, "name": "Премиум"},
}

# Обновлённые промпты с указанием только разрешённых тегов
GPT_PROMPT = (
    "Перепиши немного пост своими словами не меняя его сути, обязательно сохраняя HTML-разметку "
    "(теги <b>, <i>, <u>, <s>, <tg-spoiler>, <a>, <code>, <pre>, <blockquote>). "
    "Если в тексте есть URL-адреса, оберни их в теги <a href='URL'>URL</a>. "
    "Добавь свою разметку (например, делай заголовки жирными с помощью <b>), но сохраняй все ссылки в тегах <a> (обязательно сохраняй ссылки), "
    "кроме ссылок вида t.me/, которые находятся в конце текста. Удаляй упоминания аккаунтов (например, @username) "
    "и слова 'не баг а фича', чтобы не было телеграмм ссылок вида t.me/, также удаляй текст INCUBE.AI | ПОДПИСАТЬСЯ если он присутствует. "
    "Сохраняй исходное форматирование и структуру текста. "
    "Не используй никакие другие HTML-теги, кроме <b>, <i>, <u>, <s>, <tg-spoiler>, <a>, <code>, <pre>, <blockquote>."
)

CONDENSE_PROMPT = (
    "Сократи этот текст, сохранив основную мысль, все ссылки в тегах <a>, "
    "HTML-разметку (теги <b>, <i>, <u>, <s>, <tg-spoiler>, <a>, <code>, <pre>, <blockquote>) и структуру. "
    "Если в тексте есть URL-адреса, убедись, что они обернуты в теги <a href='URL'>URL</a>. "
    "Убедись, что ключевые детали остаются в тексте, а сокращение минимально. "
    "Не используй никакие другие HTML-теги, кроме <b>, <i>, <u>, <s>, <tg-spoiler>, <a>, <code>, <pre>, <blockquote>."
)

MAX_SYMBOLS_MESSAGE = 4096 
GPT_MAX_TOKENS = 500
GPT_TEMPERATURE = 0.7
LINK_CAPTION = "INCUBE.AI | ПОДПИСАТЬСЯ"
LINK_APPEND = f'{NL * 2}<a href="{CHANNEL_URL}">{LINK_CAPTION}</a>'
MAX_SYMBOLS_CAPTION = 1024

openai.api_key = OPENAI_API_KEY

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
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

# Модели данных
class UserStatus(Enum):
    GUEST = 0
    TRIAL = 1
    PAID = 2
    ADMIN = 3

DB_NAME = "smm_bot.db"

def init_db():
    with closing(sqlite3.connect(DB_NAME)) as conn:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    status INTEGER,
                    posts_left INTEGER,
                    channel_id TEXT,
                    tariff TEXT,
                    paid_until TEXT
                )
            """)

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# Модель User с методами для работы с БД
class User:
    def __init__(
        self,
        user_id: int,
        username: str = None,
        status: UserStatus = UserStatus.GUEST,
        posts_left: int = 0,
        channel_id: str = None,
        tariff: str = None,
        paid_until: datetime = None,
    ):
        self.user_id = user_id
        self.username = username
        self.status = status
        self.posts_left = posts_left
        self.channel_id = channel_id
        self.tariff = tariff
        self.paid_until = paid_until

    def is_active(self) -> bool:
        if self.status == UserStatus.ADMIN:
            return True
        if self.status == UserStatus.GUEST:
            return False
        if self.paid_until and datetime.now() > self.paid_until:
            return False
        return self.posts_left > 0

    def use_post(self) -> bool:
        if self.status == UserStatus.ADMIN:
            return True
        if not self.is_active():
            return False
        if self.posts_left > 0:
            self.posts_left -= 1
            self.save()
            return True
        return False

    def save(self):
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO users 
                (user_id, username, status, posts_left, channel_id, tariff, paid_until)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    self.user_id,
                    self.username,
                    self.status.value,
                    self.posts_left,
                    self.channel_id,
                    self.tariff,
                    self.paid_until.isoformat() if self.paid_until else None,
                ),
            )

    @classmethod
    def get(cls, user_id: int) -> Optional["User"]:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row:
                return cls(
                    user_id=row["user_id"],
                    username=row["username"],
                    status=UserStatus(row["status"]),
                    posts_left=row["posts_left"],
                    channel_id=row["channel_id"],
                    tariff=row["tariff"],
                    paid_until=datetime.fromisoformat(row["paid_until"])
                    if row["paid_until"]
                    else None,
                )
            return None

    @classmethod
    def create(cls, user_id: int, username: str) -> "User":
        user = cls(user_id, username, UserStatus.TRIAL, TRIAL_POSTS_LIMIT)
        user.save()
        return user

    @classmethod
    def get_all_admins(cls) -> List["User"]:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM users WHERE status = ?", (UserStatus.ADMIN.value,)
            ).fetchall()
            return [cls(**dict(row)) for row in rows]

    @classmethod
    def count_by_status(cls, status: UserStatus) -> int:
        with get_db_connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM users WHERE status = ?", (status.value,)
            ).fetchone()[0]

# База данных (SQLite)
class Database:
    def __init__(self):
        init_db()
        self._init_admins()

    def _init_admins(self):
        for admin_id in ADMIN_IDS:
            if not User.get(admin_id):
                User(
                    user_id=admin_id, status=UserStatus.ADMIN, posts_left=999999
                ).save()

    def get_user(self, user_id: int) -> User:
        return User.get(user_id) or User(user_id)

    def create_user(self, user_id: int, username: str) -> User:
        return User.create(user_id, username)

    def set_channel(self, user_id: int, channel_id: str):
        user = self.get_user(user_id)
        user.channel_id = channel_id
        user.save()

    def set_tariff(self, user_id: int, tariff: str):
        user = self.get_user(user_id)
        user.status = UserStatus.PAID
        user.tariff = tariff
        user.posts_left = TARIFFS[tariff]["posts"]
        user.paid_until = datetime.now() + timedelta(days=TARIFFS[tariff]["days"])
        user.save()

    def get_stats(self) -> Tuple[int, int, int, int]:
        with get_db_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            active = conn.execute(
                """
                SELECT COUNT(*) FROM users 
                WHERE (status = ? AND posts_left > 0) 
                OR (status = ? AND (paid_until IS NULL OR paid_until > ?))
            """,
                (
                    UserStatus.TRIAL.value,
                    UserStatus.PAID.value,
                    datetime.now().isoformat(),
                ),
            ).fetchone()[0]
            trial = User.count_by_status(UserStatus.TRIAL)
            paid = User.count_by_status(UserStatus.PAID)
            return total, active, trial, paid

db = Database()

# Определение состояний
class BotStates(StatesGroup):
    default = State()
    waiting_for_channel = State()
    waiting_for_payment = State()
    waiting_for_manual_edit = State()

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
        data: Dict[str, Any],
    ) -> Any:
        id_ = event.media_group_id
        if not id_:
            return await handler(event, data)

        if len(self.album_data.get(id_, [])):
            self.album_data[id_].append(event)
            return None

        self.album_data[id_] = [event]
        await asyncio.sleep(self.latency)

        data["album"] = self.album_data[event.media_group_id]
        return await handler(event, data)

dp.message.middleware(AlbumMiddleware())

# Вспомогательные функции
def format_text(text: str) -> str:
    """Улучшает форматирование текста, сохраняя переносы строк и убирая лишние пробелы внутри строк."""
    lines = text.split("\n")
    formatted_lines = [" ".join(line.split()) for line in lines]
    return "\n".join(formatted_lines).strip()

def clean_html_for_telegram(text: str) -> str:
    """Очищает HTML, оставляя только поддерживаемые Telegram теги включая скрытый, подчеркнутый и зачеркнутый текст."""
    if not text:
        return ""
    
    text = text.replace("\ufeff", "").replace("\u200b", "")
    soup = bs(text, "html.parser")
    
    # Полный список поддерживаемых Telegram тегов
    allowed_tags = {
        "b", "strong",           # жирный
        "i", "em",              # курсив  
        "u",                    # подчеркнутый
        "s", "strike", "del",   # зачеркнутый
        "spoiler", "tg-spoiler", # скрытый текст
        "code",                 # моноширинный
        "pre",                  # блок кода
        "a",                    # ссылка
        "blockquote"            # цитата
    }
    
    for tag in soup.find_all():
        if tag.name not in allowed_tags:
            tag.unwrap()
        else:
            if tag.name == "a":
                # Для ссылок сохраняем только href атрибут
                if "href" not in tag.attrs:
                    tag.unwrap()
                else:
                    tag.attrs = {"href": tag["href"]}
            elif tag.name in ["spoiler", "tg-spoiler"]:
                # Конвертируем в стандартный тег spoiler для Telegram
                tag.name = "tg-spoiler" 
                tag.attrs = {}
            elif tag.name in ["s", "strike", "del"]:
                # Конвертируем все варианты зачеркнутого в стандартный <s>
                tag.name = "s"
                tag.attrs = {}
            elif tag.name in ["strong"]:
                # Конвертируем <strong> в <b>
                tag.name = "b"
                tag.attrs = {}
            elif tag.name in ["em"]:
                # Конвертируем <em> в <i>
                tag.name = "i"
                tag.attrs = {}
            else:
                # Для остальных тегов убираем все атрибуты
                tag.attrs = {}
    
    cleaned_text = str(soup)
    cleaned_text = format_text(cleaned_text)
    return cleaned_text


def truncate_html(text: str, trunc: int) -> str:
    """Обрезает текст до указанной длины, сохраняя HTML-структуру."""
    soup = bs(text, "html.parser")
    current_length = 0
    for node in soup.find_all(text=True):
        if node.parent.name in ["b", "i", "a", "code", "pre"]:
            text_content = node.string
            if text_content:
                if current_length + len(text_content) > trunc:
                    allowed_length = trunc - current_length
                    node.string = text_content[:allowed_length] + "..."
                    current_length += allowed_length
                    break
                current_length += len(text_content)
    return str(soup)

async def chat_completion(
    prompt: str, custom_prompt: Optional[str] = None
) -> Dict[str, Union[bool, str]]:
    if not prompt:
        return {"ok": True, "text": "", "error": ""}
    system_message = custom_prompt if custom_prompt else GPT_PROMPT
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            max_tokens=GPT_MAX_TOKENS,
            temperature=GPT_TEMPERATURE,
        )
        msg = response.choices[0].message["content"]
        return {"ok": True, "text": msg.strip() if msg else "", "error": ""}
    except Exception as e:
        logging.error(f"Error in chat_completion: {e}")
        return {"ok": False, "text": "", "error": str(e)}

async def rewrite(
    text_to_send: str,
    trunc: Optional[int] = MAX_SYMBOLS_CAPTION,
    custom_prompt: Optional[str] = None,
) -> Dict[str, Union[bool, str]]:
    if not text_to_send:
        return {"ok": True, "text": "", "error": ""}

    full_prompt = custom_prompt if custom_prompt else GPT_PROMPT

    # Первый этап: модификация текста
    result1 = await chat_completion(text_to_send, custom_prompt=full_prompt)
    if not result1["ok"]:
        return result1

    modified_text = result1["text"].strip()
    if not modified_text:
        return {
            "ok": False,
            "text": "",
            "error": "OpenAI вернул пустой ответ на первом этапе.",
        }

    # Второй этап: сокращение текста (если текст длиннее лимита)
    final_text = modified_text
    if trunc and len(modified_text) > trunc - len(LINK_APPEND):
        result2 = await chat_completion(modified_text, custom_prompt=CONDENSE_PROMPT)
        if not result2["ok"]:
            return result2
        final_text = result2["text"].strip()
        if not final_text:
            return {
                "ok": False,
                "text": "",
                "error": "OpenAI вернул пустой ответ на втором этапе.",
            }

    # Добавляем ссылку только если её еще нет
    if trunc:
        # Сначала добавляем ссылку, затем обрезаем
        final_text_with_link = add_link_once(final_text)
        if len(final_text_with_link) > trunc:
            # Если превышает лимит, обрезаем основной текст
            max_content_length = trunc - len(LINK_APPEND)
            truncated = truncate_html(final_text, max_content_length)
            final_text = add_link_once(truncated)
        else:
            final_text = final_text_with_link
    else:
        final_text = add_link_once(final_text)

    final_text = clean_html_for_telegram(final_text)
    return {"ok": True, "text": final_text, "error": ""}


# Функция для добавления ссылки только один раз
def add_link_once(text: str) -> str:
    """Добавляет ссылку на канал только если её еще нет в тексте."""
    if not text:
        return LINK_APPEND.strip()
    
    # Проверяем, есть ли уже ссылка в тексте
    if LINK_CAPTION in text or CHANNEL_URL in text:
        return text
    
    return f"{text}{LINK_APPEND}"

# Клавиатуры
def get_preview_keyboard(allow_edit=True):
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="Отправить", callback_data="confirm"))
    if allow_edit:
        builder.add(InlineKeyboardButton(text="Редактировать вручную", callback_data="edit_manual"))
        builder.add(InlineKeyboardButton(text="Сгенерировать заново", callback_data="regenerate"))
    builder.add(InlineKeyboardButton(text="Отменить", callback_data="cancel"))
    return builder.as_markup()

def get_tariffs_keyboard():
    builder = InlineKeyboardBuilder()
    for tariff_id, tariff in TARIFFS.items():
        builder.add(
            InlineKeyboardButton(
                text=f"{tariff['name']} - {tariff['price']} руб.",
                callback_data=f"tariff_{tariff_id}",
            )
        )
    builder.add(InlineKeyboardButton(text="Отмена", callback_data="cancel_payment"))
    builder.adjust(1)
    return builder.as_markup()

preview_keyboard = get_preview_keyboard(allow_edit=True)
album_preview_keyboard = get_preview_keyboard(allow_edit=False)
tariffs_keyboard = get_tariffs_keyboard()

# Функция отправки превью с обработкой ошибок
async def send_preview(message: types.Message, state: FSMContext):
    data = await state.get_data()
    preview_data = data.get("preview_data")
    if not preview_data:
        await message.edit_text("❌ Ошибка: данные для превью отсутствуют.")
        return
    message_type = preview_data["type"]
    try:
        if message_type == "text":
            preview_msg = await bot.send_message(
                message.chat.id,
                preview_data["text"],
                parse_mode=ParseMode.HTML,
                reply_markup=preview_keyboard,
            )
        elif message_type in ["photo", "video", "animation", "document"]:
            if message_type == "photo":
                preview_msg = await bot.send_photo(
                    message.chat.id,
                    preview_data["media"],
                    caption=preview_data.get("caption"),
                    parse_mode=ParseMode.HTML,
                    reply_markup=preview_keyboard,
                )
            elif message_type == "video":
                preview_msg = await bot.send_video(
                    message.chat.id,
                    preview_data["media"],
                    caption=preview_data.get("caption"),
                    parse_mode=ParseMode.HTML,
                    reply_markup=preview_keyboard,
                )
            elif message_type == "animation":
                preview_msg = await bot.send_animation(
                    message.chat.id,
                    preview_data["media"],
                    caption=preview_data.get("caption"),
                    parse_mode=ParseMode.HTML,
                    reply_markup=preview_keyboard,
                )
            elif message_type == "document":
                preview_msg = await bot.send_document(
                    message.chat.id,
                    preview_data["media"],
                    caption=preview_data.get("caption"),
                    parse_mode=ParseMode.HTML,
                    reply_markup=preview_keyboard,
                )
        elif message_type == "media_group":
            media_group = []
            for item in preview_data["media"]:
                if item["type"] == "photo":
                    media_group.append(
                        types.InputMediaPhoto(
                            media=item["media"],
                            caption=item.get("caption"),
                            parse_mode=item.get("parse_mode"),
                        )
                    )
                elif item["type"] == "video":
                    media_group.append(
                        types.InputMediaVideo(
                            media=item["media"],
                            caption=item.get("caption"),
                            parse_mode=item.get("parse_mode"),
                        )
                    )
            await bot.send_media_group(message.chat.id, media_group)
            preview_msg = await message.answer(
                "Превью альбома выше", reply_markup=album_preview_keyboard
            )
        await state.update_data(preview_message_id=preview_msg.message_id)
    except TelegramBadRequest as e:
        logging.error(f"Ошибка отправки сообщения: {e}, текст: {preview_data.get('caption') or preview_data.get('text')}")
        await message.edit_text("❌ Ошибка при отправке превью: неверный формат текста. Попробуйте снова.")
        return

# Хендлеры команд
@dp.message(CommandStart())
async def start_handler(message: types.Message, state: FSMContext):
    user = db.get_user(message.from_user.id)

    if user.status == UserStatus.GUEST:
        db.create_user(message.from_user.id, message.from_user.username)
        await message.answer(
            f"👋 Добро пожаловать, {message.from_user.full_name}!\n\n"
            f"🎉 Вы получили {TRIAL_POSTS_LIMIT} бесплатных постов. "
            "Для продолжения работы привяжите свой канал с помощью команды /set_channel."
        )
    else:
        status_msg = (
            "администратор"
            if user.status == UserStatus.ADMIN
            else "платный пользователь"
            if user.status == UserStatus.PAID
            else "пробный пользователь"
        )
        posts_left = user.posts_left if user.posts_left > 0 else 0
        if user.status == UserStatus.ADMIN:
            posts_left = "неограничено"
        await message.answer(
            f"👋 С возвращением, {message.from_user.full_name}!\n\n"
            f"📊 Ваш статус: {status_msg}\n"
            f"📨 Осталось постов: {posts_left}\n"
            f"Команды:\n"
            f"Установить новый канал: /set_channel\n"
            f"Посмотреть статистику: /stats\n"
            "Отправьте мне пост, и я помогу его оформить для публикации."
        )
    await state.set_state(BotStates.default)

@dp.message(Command("set_channel"))
async def set_channel_handler(message: types.Message, state: FSMContext):
    user = db.get_user(message.from_user.id)
    if user.status == UserStatus.GUEST:
        await message.answer("❌ Сначала зарегистрируйтесь с помощью команды /start")
        return
    await message.answer(
        "📢 Пришлите @username или ID вашего канала. "
        "Бот должен быть администратором в этом канале."
    )
    await state.set_state(BotStates.waiting_for_channel)

@dp.message(BotStates.waiting_for_channel)
async def process_channel(message: types.Message, state: FSMContext):
    channel_id = message.text.strip()
    if not channel_id:
        await message.answer("❌ Неверный формат канала. Попробуйте еще раз.")
        return
    db.set_channel(message.from_user.id, channel_id)
    await message.answer(f"✅ Канал {channel_id} успешно привязан!")
    await state.set_state(BotStates.default)

@dp.message(Command("buy"))
async def buy_handler(message: types.Message, state: FSMContext):
    user = db.get_user(message.from_user.id)
    if user.status == UserStatus.GUEST:
        await message.answer("❌ Сначала зарегистрируйтесь с помощью команды /start")
        return
    await message.answer(
        "💰 Выберите тарифный план:\n\n"
        f"🔹 {TARIFFS['basic']['name']} - {TARIFFS['basic']['posts']} постов / {TARIFFS['basic']['price']} руб.\n"
        f"🔸 {TARIFFS['standard']['name']} - {TARIFFS['standard']['posts']} постов / {TARIFFS['standard']['price']} руб.\n"
        f"🔺 {TARIFFS['premium']['name']} - {TARIFFS['premium']['posts']} постов / {TARIFFS['premium']['price']} руб.",
        reply_markup=tariffs_keyboard,
    )
    await state.set_state(BotStates.waiting_for_payment)

@dp.callback_query(F.data.startswith("tariff_"))
async def tariff_selected(callback: types.CallbackQuery, state: FSMContext):
    tariff_id = callback.data.split("_")[1]
    if tariff_id not in TARIFFS:
        await callback.answer("❌ Неверный тариф")
        return
    tariff = TARIFFS[tariff_id]
    await callback.message.edit_text(
        f"✅ Вы выбрали тариф {tariff['name']}\n\n"
        f"💳 Сумма к оплате: {tariff['price']} руб.\n\n"
        "Для оплаты переведите указанную сумму на наш счет и отправьте скриншот чека.",
        reply_markup=None,
    )
    db.set_tariff(callback.from_user.id, tariff_id)
    await callback.answer(f"Тариф {tariff['name']} активирован!")
    await state.set_state(BotStates.default)

@dp.message(Command("stats"))
async def stats_handler(message: types.Message):
    user = db.get_user(message.from_user.id)
    if user.status != UserStatus.ADMIN:
        await message.answer("❌ Эта команда доступна только администраторам")
        return
    total_users, active_users, trial_users, paid_users = db.get_stats()
    stats_msg = "📊 Статистика бота:\n\n"
    stats_msg += f"👥 Всего пользователей: {total_users}\n"
    stats_msg += f"🟢 Активных: {active_users}\n"
    stats_msg += f"🟡 Пробных: {trial_users}\n"
    stats_msg += f"🔵 Платных: {paid_users}\n\n"
    await message.answer(stats_msg)

# Основной обработчик сообщений
@dp.message(BotStates.default, ~F.media_group_id)
async def message_handler(message: types.Message, state: FSMContext):
    user = db.get_user(message.from_user.id)
    if not user.is_active():
        await message.answer(
            "❌ У вас закончились доступные посты или подписка.\n\n"
            "🔄 Для продления используйте команду /buy"
        )
        return
    is_text = message.content_type == ContentType.TEXT
    text_to_send = message.html_text or message.caption
    if not text_to_send:
        return
    await message.answer("⏳ Обрабатываю ваше сообщение...")
    rewrite_result = await rewrite(
        text_to_send=text_to_send,
        trunc=MAX_SYMBOLS_MESSAGE if is_text else MAX_SYMBOLS_CAPTION,
    )
    if not rewrite_result["ok"]:
        await message.answer(f"❌ Ошибка: {rewrite_result['error']}")
        return
    final_text = rewrite_result["text"].strip()
    if not final_text:
        await message.answer("❌ OpenAI вернул пустой ответ.")
        return
    await message.answer("✅ Текст сгенерирован, отправляю превью...")
    preview_data = {
        "type": "text" if is_text else message.content_type,
        "text": final_text if is_text else None,
        "caption": final_text if not is_text else None,
        "media": (
            message.photo[-1].file_id
            if message.photo
            else message.video.file_id
            if message.video
            else message.animation.file_id
            if message.animation
            else message.document.file_id
            if message.document
            else None
        ),
    }
    await state.update_data(preview_data=preview_data, original_text=text_to_send)
    await send_preview(message, state)

@dp.message(F.media_group_id)
async def album_handler(
    message: types.Message, album: list[types.Message], state: FSMContext
):
    user = db.get_user(message.from_user.id)
    if not user.is_active():
        await message.answer(
            "❌ У вас закончились доступные посты или подписка.\n\n"
            "🔄 Для продления используйте команду /buy"
        )
        return
    await message.answer("⏳ Обрабатываю ваш альбом...")
    captions = [obj.caption or obj.html_text for obj in album if obj.caption or obj.html_text]
    combined_caption = "\n".join(captions)
    rewrite_result = await rewrite(combined_caption, MAX_SYMBOLS_CAPTION)
    if not rewrite_result["ok"]:
        await message.answer(f"❌ Ошибка: {rewrite_result['error']}")
        return
    final_caption = rewrite_result["text"].strip()
    if not final_caption:
        final_caption = LINK_APPEND.strip()
    media_list = []
    for i, obj in enumerate(album):
        media = getattr(obj, obj.content_type, None)
        if not media:
            continue
        if obj.content_type == ContentType.PHOTO:
            media = media[-1]
        file_id = getattr(media, "file_id", None)
        if not file_id:
            continue
        caption = final_caption if i == 0 else None
        media_list.append(
            {
                "type": obj.content_type,
                "media": file_id,
                "caption": caption,
                "parse_mode": ParseMode.HTML if caption else None,
            }
        )
    await message.answer("✅ Альбом обработан, отправляю превью...")
    await state.update_data(preview_data={"type": "media_group", "media": media_list})
    await send_preview(message, state)

# Обработчики кнопок
@dp.callback_query(F.data == "confirm")
async def confirm_handler(callback: types.CallbackQuery, state: FSMContext):
    user = db.get_user(callback.from_user.id)
    if not user.use_post():
        await callback.message.edit_text(
            "❌ У вас закончились доступные посты.\n\n"
            "🔄 Для продления используйте команду /buy"
        )
        return
    data = await state.get_data()
    preview_data = data.get("preview_data")
    if not preview_data:
        await callback.message.edit_text("❌ Ошибка: данные для отправки отсутствуют.")
        return
    channel_id = user.channel_id or CHANNEL_ID
    if not channel_id:
        await callback.message.edit_text(
            "❌ Вы не можете отправить сообщение, так как не привязали свой тг канал! Используйте эту команду для привязки: /set_channel"
        )
        return
    message_type = preview_data["type"]
    try:
        if message_type == "text":
            await bot.send_message(
                channel_id, preview_data["text"], parse_mode=ParseMode.HTML
            )
        elif message_type in ["photo", "video", "animation", "document"]:
            if message_type == "photo":
                await bot.send_photo(
                    channel_id,
                    preview_data["media"],
                    caption=preview_data.get("caption"),
                    parse_mode=ParseMode.HTML,
                )
            elif message_type == "video":
                await bot.send_video(
                    channel_id,
                    preview_data["media"],
                    caption=preview_data.get("caption"),
                    parse_mode=ParseMode.HTML,
                )
            elif message_type == "animation":
                await bot.send_animation(
                    channel_id,
                    preview_data["media"],
                    caption=preview_data.get("caption"),
                    parse_mode=ParseMode.HTML,
                )
            elif message_type == "document":
                await bot.send_document(
                    channel_id,
                    preview_data["media"],
                    caption=preview_data.get("caption"),
                    parse_mode=ParseMode.HTML,
                )
        elif message_type == "media_group":
            media_group = []
            for item in preview_data["media"]:
                if item["type"] == "photo":
                    media_group.append(
                        types.InputMediaPhoto(
                            media=item["media"],
                            caption=item.get("caption"),
                            parse_mode=item.get("parse_mode"),
                        )
                    )
                elif item["type"] == "video":
                    media_group.append(
                        types.InputMediaVideo(
                            media=item["media"],
                            caption=item.get("caption"),
                            parse_mode=item.get("parse_mode"),
                        )
                    )
            await bot.send_media_group(channel_id, media_group)
        await callback.message.edit_reply_markup(reply_markup=None)
        await callback.message.answer("✅ Сообщение отправлено в канал.")
    except Exception as e:
        logging.error(f"Ошибка при отправке в канал: {e}")
        await callback.message.answer(
            "❌ Ошибка при отправке в канал. Проверьте права бота.", reply_markup=None
        )
    await state.set_state(BotStates.default)

@dp.callback_query(F.data == "edit_manual")
async def edit_manual_handler(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("Отправьте отредактированный текст:")
    await state.set_state(BotStates.waiting_for_manual_edit)


# Обновленная функция обработки ручного редактирования
@dp.message(BotStates.waiting_for_manual_edit)
async def process_manual_edit(message: types.Message, state: FSMContext):
    # Получаем текст с полной разметкой через html_text
    edited_text = message.html_text
    
    if not edited_text:
        await message.answer("❌ Пустое сообщение. Попробуйте снова.")
        return
    
    # Используем расширенную функцию очистки
    cleaned_text = clean_html_for_telegram(edited_text)
    
    data = await state.get_data()
    preview_data = data.get("preview_data")
    
    if not preview_data:
        await message.answer("❌ Ошибка: данные для редактирования отсутствуют.")
        return
    
    # Добавляем ссылку на канал только если её еще нет
    final_text = add_link_once(cleaned_text)
    
    if preview_data["type"] == "text":
        preview_data["text"] = final_text
    else:
        preview_data["caption"] = final_text
    
    await state.update_data(preview_data=preview_data)
    await message.answer("✅ Текст отредактирован, отправляю обновленное превью...")
    await send_preview(message, state)



@dp.callback_query(F.data == "regenerate")
async def regenerate_handler(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.edit_reply_markup(reply_markup=None)
    data = await state.get_data()
    original_text = data.get("original_text")
    if not original_text:
        await callback.message.answer("❌ Оригинальный текст отсутствует.")
        return
    await callback.message.answer("⏳ Генерирую текст заново...")
    rewrite_result = await rewrite(original_text)
    if not rewrite_result["ok"]:
        await callback.message.answer(f"❌ Ошибка: {rewrite_result['error']}")
        return
    final_text = rewrite_result["text"].strip()
    if not final_text:
        await callback.message.answer("❌ OpenAI вернул пустой ответ.")
        return
    preview_data = data.get("preview_data")
    if preview_data["type"] == "text":
        preview_data["text"] = final_text
    else:
        preview_data["caption"] = final_text
    await state.update_data(preview_data=preview_data)
    await send_preview(callback.message, state)

@dp.callback_query(F.data == "cancel")
async def cancel_handler(callback: types.CallbackQuery, state: FSMContext):
    if callback.message.text:
        await callback.message.edit_text("❌ Действие отменено.", reply_markup=None)
    elif callback.message.caption:
        await callback.message.edit_caption(caption="❌ Действие отменено.", reply_markup=None)
    else:
        await callback.message.delete()
        await callback.message.answer("❌ Действие отменено.")
    await state.set_state(BotStates.default)

@dp.callback_query(F.data == "cancel_payment")
async def cancel_payment_handler(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.edit_text("❌ Выбор тарифа отменен.", reply_markup=None)
    await state.set_state(BotStates.default)

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

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот выключен.")