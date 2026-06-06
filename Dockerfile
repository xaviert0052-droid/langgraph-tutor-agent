FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（FAISS + scikit-learn 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制项目
COPY pyproject.toml .
COPY src/ src/
COPY webapp/ webapp/

# 安装 Python 依赖
RUN pip install --no-cache-dir -e .

# 暴露端口
EXPOSE 5000

# 启动（用 gunicorn 生产级服务器）
CMD gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 webapp.server:app
