# -------- Stage 1: builder (ставим Python, torch, зависимости) --------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Moscow

# Базовые пакеты + Python 3.11
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        python3.11 python3.11-distutils python3-pip \
        libopenblas0 liblapack3 libx11-6 libgtk-3-0 libgl1 \
        ffmpeg && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip

WORKDIR /app

# Сначала PyTorch с CUDA 11.8
RUN python3.11 -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Затем зависимости проекта
COPY requirements.txt .
RUN python3.11 -m pip install -r requirements.txt

# Очистка кеша pip
RUN python3.11 -m pip cache purge

# -------- Stage 2: runtime (минимальный образ для запуска) --------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Moscow

# Системные либы + ffmpeg (для OpenCV)
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y --no-install-recommends \
        tzdata software-properties-common \
        build-essential cmake \
        libopenblas-dev liblapack-dev \
        libx11-dev libgtk-3-dev libgl1 \
        python3.11 python3.11-dev python3.11-distutils python3-pip && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip

WORKDIR /app

# Копируем установленный PyTorch и остальные пакеты из builder
COPY --from=builder /usr/local /usr/local


# Образ содержит только систему, Python, torch, зависимости.
# Код монтируется томом, поэтому внутрь на этапе сборки не копируется.
# Но если нужно собрать "самодостаточный" образ, раскомментировать следующую строку:
# COPY . .
