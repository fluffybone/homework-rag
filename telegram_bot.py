import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import pickle
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
INDEX_PATH = "/app/index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "IlyaGusev/fred_t5_ru_turbo_small"  # Легкая модель для русского

class TelegramRAGBot:
    def __init__(self):
        print("🤖 Инициализация RAG бота...")
        
        # Загрузка модели для эмбеддингов
        print("📥 Загрузка модели эмбеддингов...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Загрузка FAISS индекса
        print("📂 Загрузка векторного индекса...")
        self.index = faiss.read_index(os.path.join(INDEX_PATH, "index.faiss"))
        
        # Загрузка метаданных
        with open(os.path.join(INDEX_PATH, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        
        # Загрузка LLM модели
        print("🧠 Загрузка языковой модели...")
        self.llm = self.load_llm()
        
        print("✅ Бот готов к работе!")

    def load_llm(self):
        """Загрузка легкой LLM модели"""
        try:
            return pipeline(
                "text2text-generation",
                model=LLM_MODEL,
                max_length=512,
                torch_dtype=torch.float32,
                device=-1  # CPU
            )
        except Exception as e:
            print(f"❌ Ошибка загрузки LLM: {e}")
            print("🔄 Будет использован режим только поиска")
            return None

    def search_documents(self, query, k=3):
        """Поиск релевантных документов"""
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        query_embedding = query_embedding.astype('float32')
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    'text': self.metadata[idx]['text'],
                    'source': self.metadata[idx]['source'],
                    'distance': distances[0][i]
                })
        
        return results

    def generate_answer(self, question, context):
        """Генерация ответа с помощью LLM"""
        if self.llm is None:
            return f"На основе документов:\n\n{context}\n\nОтвет на вопрос: {question}"
        
        prompt = f"""
Ответь на вопрос на основе предоставленного контекста. Если ответа нет в контексте, скажи "Не знаю".

Контекст:
{context}

Вопрос: {question}

Ответ:
"""
        try:
            result = self.llm(
                prompt,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True
            )
            return result[0]['generated_text'].replace(prompt, "").strip()
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            return f"На основе документов:\n\n{context}\n\nОтвет на вопрос: {question}"

    def ask_question(self, question):
        """Ответ на вопрос"""
        # Поиск релевантных документов
        results = self.search_documents(question, k=3)
        
        if not results:
            return "❌ Не найдено подходящей информации в документах.", []

        # Формирование контекста
        context = "\n\n".join([result['text'] for result in results])
        sources = list(set([result['source'] for result in results]))

        # Генерация ответа с помощью LLM
        answer = self.generate_answer(question, context)
        
        # Форматируем ответ для Telegram
        formatted_answer = f"🔍 *Ответ на ваш вопрос:*\n\n"
        formatted_answer += f"{answer}\n\n"
        formatted_answer += f"📚 *Источники:* {', '.join(sources)}"
        
        return formatted_answer, sources

# Инициализация бота
rag_bot = TelegramRAGBot()

# Команды бота
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = """
👋 *Добро пожаловать в RAG бота с ИИ!*

Я могу искать информацию в ваших документах и генерировать ответы с помощью нейросети.

📝 *Как использовать:*
• Просто напишите ваш вопрос
• Я найду relevant информацию
• Сгенерирую ответ с помощью ИИ
• Покажу ответ с источниками

🎯 *Примеры вопросов:*
• Что известно о Петре Пупкине?
• Расскажи о проекте X
• Какие навыки у сотрудника Y

Напишите ваш вопрос ниже 👇
"""
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
📖 *Помощь по использованию бота*

*Команды:*
/start - начать работу с ботом
/help - показать эту справку
/about - информация о боте

*Как работает:*
1. Вы задаете вопрос на русском языке
2. Я ищу информацию в базе документов
3. Нейросеть генерирует ответ на основе найденной информации
4. Показываю ответ с указанием источников

🤖 *Технологии:*
• FAISS для векторного поиска
• Sentence Transformers для эмбеддингов
• ИИ модель для генерации ответов
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /about"""
    about_text = """
🤖 *О боте*

*RAG (Retrieval Augmented Generation) бот с ИИ*
• Семантический поиск по документам
• Генерация ответов нейросетью
• Русскоязычные эмбеддинги
• Локальное исполнение

*Используемые модели:*
• paraphrase-multilingual-MiniLM-L12-v2 - для эмбеддингов
• fred_t5_ru_turbo_small - для генерации ответов

📊 *База знаний:*
• Обработано документов: 5
• Размер базы: ~58 фрагментов
"""
    await update.message.reply_text(about_text, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    try:
        user_message = update.message.text
        
        # Показываем статус "печатает..."
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, 
            action="typing"
        )
        
        # Получаем ответ от RAG системы
        answer, sources = rag_bot.ask_question(user_message)
        
        # Отправляем ответ пользователю
        await update.message.reply_text(answer, parse_mode='Markdown')
        
        logger.info(f"User: {update.effective_user.id}, Question: {user_message}")
        
    except Exception as e:
        error_text = "❌ Произошла ошибка при обработке запроса. Попробуйте еще раз."
        await update.message.reply_text(error_text)
        logger.error(f"Error: {e}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ Произошла непредвиденная ошибка. Попробуйте еще раз."
        )

def main():
    """Запуск Telegram бота"""
    # Получаем токен из переменных окружения
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ TELEGRAM_BOT_TOKEN not found in environment variables!")
        print("❌ Создайте файл .env с TELEGRAM_BOT_TOKEN=ваш_токен")
        return
    
    # Проверяем наличие индекса
    if not os.path.exists(INDEX_PATH) or not os.listdir(INDEX_PATH):
        print("📭 Индекс не найден! Сначала запустите:")
        print("python create_index.py")
        return
    
    # Создаем приложение
    application = Application.builder().token(token).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    
    # Запускаем бота
    print("🤖 Telegram бот запускается...")
    print("✅ Бот готов принимать сообщения!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()