import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI # Используем для интеграции с OpenAI-compatible API (OpenRouter)
from langchain_core.documents import Document

# --- Настройка логирования ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Загрузка переменных окружения ---
load_dotenv()

# --- Конфигурация ---
# Получаем переменные из .env файла
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "index/knowledge_index") # Дефолтный путь, если не в .env
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base") # Дефолтный путь

# Проверка наличия необходимых ключей
if not OPENAI_API_KEY or not TELEGRAM_BOT_TOKEN:
    logger.error("OPENAI_API_KEY или TELEGRAM_BOT_TOKEN не установлены. Проверьте ваш .env файл.")
    exit(1)

# --- Инициализация моделей ---
# Модель эмбеддингов
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    logger.info(f"Инициализирована модель эмбеддингов: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Ошибка при инициализации модели эмбеддингов {EMBEDDING_MODEL_NAME}: {e}")
    exit(1)

# LLM модель через OpenRouter
try:
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        temperature=0.7, # Параметр креативности ответа
    )
    logger.info(f"Инициализирована LLM модель: {LLM_MODEL_NAME} через {OPENAI_API_BASE}")
except Exception as e:
    logger.error(f"Ошибка при инициализации LLM {LLM_MODEL_NAME}: {e}")
    exit(1)


# --- Функции для работы с базой знаний и FAISS ---

def load_and_split_documents(knowledge_base_dir: str):
    """Загружает текстовые файлы из директории и возвращает список документов LangChain."""
    documents = []
    logger.info(f"Загрузка документов из директории: {knowledge_base_dir}")
    if not os.path.isdir(knowledge_base_dir):
        logger.warning(f"Директория базы знаний '{knowledge_base_dir}' не найдена.")
        return []

    for filename in os.listdir(knowledge_base_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(knowledge_base_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Создаем объект Document для каждого файла
                    # Можно добавить больше метаданных при необходимости
                    documents.append(Document(page_content=content, metadata={"source": filename}))
                    logger.info(f"Загружен документ: {filename}")
            except Exception as e:
                logger.error(f"Ошибка при чтении файла {filename}: {e}")
    return documents

def create_or_load_faiss_index(index_path: str, knowledge_base_dir: str, embeddings_model):
    """
    Создает FAISS индекс из документов, если он не существует,
    или загружает существующий индекс.
    """
    if os.path.exists(index_path) and FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True): # Проверка существования индекса
        logger.info(f"Загрузка существующего FAISS индекса из: {index_path}")
        try:
            vector_store = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
            return vector_store
        except Exception as e:
            logger.error(f"Ошибка при загрузке FAISS индекса: {e}. Будет создан новый.")
            # Если загрузка не удалась, продолжим создание нового
    
    logger.info("FAISS индекс не найден или поврежден. Создание нового индекса...")
    documents = load_and_split_documents(knowledge_base_dir)
    
    if not documents:
        logger.warning("Нет документов для создания индекса. Убедитесь, что файлы .txt находятся в директории knowledge_base.")
        return None # Возвращаем None, если документов нет

    # FAISS требует, чтобы документы были разделены на чанки.
    # В данном случае, мы загрузили содержимое файла как один Document.
    # Если файлы большие, вам может понадобиться RecursiveCharacterTextSplitter.
    # Для простоты, предполагаем, что каждый .txt файл - это один документ.
    # Если нужно разделение:
    # from langchain.text_splitter import RecursiveCharacterTextSplitter
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # docs = text_splitter.split_documents(documents)
    
    try:
        vector_store = FAISS.from_documents(documents, embeddings_model)
        logger.info(f"Создание FAISS индекса и сохранение в: {index_path}")
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        vector_store.save_local(index_path)
        logger.info("FAISS индекс успешно создан и сохранен.")
        return vector_store
    except Exception as e:
        logger.error(f"Ошибка при создании FAISS индекса: {e}")
        return None

# --- Создание RAG цепочки ---
template = """
Ответь на вопрос, основываясь только на предоставленном контексте.
Если ты не знаешь ответа, скажи, что не знаешь. Не пытайся выдумать ответ.
Используй русский язык.

Контекст:
{context}

Вопрос:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Функции Telegram бота ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение при команде /start."""
    await update.message.reply_text('Привет! Я твой RAG-бот. Задай мне вопрос по базе знаний.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текстовые сообщения от пользователя."""
    user_question = update.message.text
    logger.info(f"Получен вопрос от пользователя: '{user_question}'")

    # Получаем или создаем FAISS индекс
    vector_store = create_or_load_faiss_index(FAISS_INDEX_PATH, KNOWLEDGE_BASE_DIR, embeddings)

    if vector_store is None:
        await update.message.reply_text("Не удалось загрузить или создать базу знаний. Пожалуйста, проверьте настройки.")
        return

    # Поиск релевантных документов в FAISS
    try:
        # Используем retriever для получения контекста
        retriever = vector_store.as_retriever()
        relevant_docs = retriever.invoke(user_question)
        context_str = format_docs(relevant_docs)
        logger.debug(f"Найденный контекст: {context_str}")

        # Создание цепочки для ответа на вопрос с учетом контекста
        # Пропускаем контекст через format_docs, а вопрос напрямую
        chain_input = {"context": context_str, "question": user_question}
        
        # Генерируем ответ от LLM
        response = rag_chain.invoke(chain_input)
        
        await update.message.reply_text(response)
        logger.info(f"Отправлен ответ пользователю: '{response[:50]}...'")

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса '{user_question}': {e}")
        await update.message.reply_text("Произошла ошибка при обработке вашего запроса.")

def main() -> None:
    """Запускает Telegram бота."""
    logger.info("Запуск Telegram бота...")
    # Создаем приложение бота
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Добавляем обработчики команд
    application.add_handler(CommandHandler("start", start))

    # Добавляем обработчик сообщений (текст, не команды)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запускаем бота
    logger.info("Бот запущен. Ожидание сообщений...")
    application.run_polling()

if __name__ == "__main__":
    # Перед первым запуском (или если файлов в knowledge_base нет)
    # можно вызвать создание индекса один раз, чтобы он был готов
    # Но лучше, если это делается динамически при первом запросе или запуске
    
    # Пример: вызвать индексацию при старте, если нужно
    # logger.info("Предварительная индексация базы знаний...")
    # create_or_load_faiss_index(FAISS_INDEX_PATH, KNOWLEDGE_BASE_DIR, embeddings)
    
    main()