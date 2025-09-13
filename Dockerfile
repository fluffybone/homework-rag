# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
# --no-cache-dir предотвращает сохранение кэша pip, уменьшая размер образа
# --upgrade pip гарантирует использование последней версии pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем остальной код приложения
COPY . .

# Создаем директории, если они не существуют (хотя docker-compose их монтирует, это хорошая практика)
RUN mkdir -p /app/knowledge_base /app/index /root/.cache/huggingface

# Команда для запуска приложения при старте контейнера
# Убедитесь, что ваш main.py экспортирует функцию или класс, который будет запущен
CMD ["python", "main.py"]