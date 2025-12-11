FROM python:3.10-slim

WORKDIR /app

# Отключаем буферизацию, чтобы логи были видны сразу
ENV PYTHONUNBUFFERED=1

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем и устанавливаем зависимости (для кэширования слоев Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем содержимое папки src внутрь контейнера в папку /app
COPY src/ .

# Запускаем скрипт (он находится в /app)
CMD ["python", "main.py"]
