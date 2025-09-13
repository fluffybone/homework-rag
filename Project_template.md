
## Добавляем данные
1) создание папки `knowledge_base` - с нашей базой данных
2) создание файда `terms_map.json`- ключевые замененные слова

# Модель для эмбеддингов ( HuggingFaceEmbeddings )
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# LLM модель для ответа (через OpenRouter)
Пробовала
LLM_MODEL_NAME=deepseek/deepseek-chat-v3.1:free (неплохо ищет но не воспринимал prompt)
и
LLM_MODEL_NAME=qwen/qwen3-coder:free - смог использовать prompt чтобы ответить по шагам 

>в конце уже в консоле была ошибка что закончились лимиты бесплатные 

## Собираем образ
docker-compose build

## Запускаем бота
docker-compose up


