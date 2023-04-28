# Использую базовый образ Python 3.9
FROM python:3.9-slim

# Установиваю необходимые зависимости
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt /app_detect/requirements.txt
RUN pip install --no-cache-dir -r /app_detect/requirements.txt

# Копирую файлы predict.py и detecting.py в контейнер
COPY predict.py /app_detect/
COPY detecting.py /app_detect/

# Установиваю директорию приложения по умолчанию
WORKDIR /app_detect

# Запускаю скрипт detection.py при запуске контейнера
CMD ["python", "detecting.py"]