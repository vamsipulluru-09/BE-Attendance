FROM python:3.9.18-slim AS builder

WORKDIR /app

COPY requirements.txt .

# Install build dependencies and optimize memory usage
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables to optimize memory usage
ENV MAKEFLAGS="-j4"
ENV CFLAGS="-O2"
ENV CXXFLAGS="-O2"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.9.18-slim

RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libhdf5-dev \
    libhdf5-serial-dev \
    libharfbuzz0b \
    libwebp6 \
    libtiff5 \
    libjasper1 \
    libopenexr23 \
    libgstreamer1.0-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libcblas3 \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY ./app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]