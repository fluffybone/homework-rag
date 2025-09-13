1) установить Питон
2) py -m venv .venv
3) . .venv\Scripts\Activate.ps1
4) pip install langchain faiss-cpu 

5) создание папки `knowledge_base` 
6) создание файда `terms_map.json`


# Запуск

# 1. Создать индекс
docker-compose run rag-bot python create_index.py

# 2. Запустить Telegram бота
docker-compose up -d

# 3. Проверить логи
docker-compose logs -f


## Собираем образ
docker-compose build

## Создаем векторный индекс
docker-compose run rag-bot python create_index.py

## Запускаем бота
docker-compose up



какие то проблемы были с совместимостью langchain и  HuggingFaceEmbeddings (ничего не понятно в общем), ну аи посоветовал вообще без нее 