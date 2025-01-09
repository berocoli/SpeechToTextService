# Python 3.11 Slim İmajı
FROM python:3.11-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1-dev \
    libffi-dev \
    git \
    && apt-get clean

# Gereksinim dosyasını kopyala
COPY requirements.txt ./

# Git bağımlılıklarını manuel olarak indir ve yükle
RUN git clone https://github.com/huggingface/datasets.git /tmp/datasets && \
    pip install -e /tmp/datasets && \
    git clone https://github.com/timonweb/django-tailwind.git /tmp/django-tailwind && \
    pip install -e /tmp/django-tailwind && \
    git clone https://github.com/SeaBenSea/HuBERT-SER.git hubert-ser
# Kalan bağımlılıkları yükle (git bağımlılıkları yorumlandığı için sorun yaratmaz)
RUN pip install --no-cache-dir -r requirements.txt || true

# Ortam değişkenlerini kopyala
COPY .env /app/.env

# Proje dosyalarını kopyala
COPY . .

# Statik dosyaları toplama (opsiyonel)
RUN python StoTextProj/manage.py collectstatic --noinput

# Portu aç (Django'nun varsayılan portu 8000)
EXPOSE 8000

# Uygulamayı çalıştır
CMD ["python", "StoTextProj/manage.py", "runserver", "0.0.0.0:8000"]
