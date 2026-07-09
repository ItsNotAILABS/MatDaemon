FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY matdaemon ./matdaemon
RUN python -m pip install --upgrade pip && python -m pip install .[api]

EXPOSE 8000
CMD ["matdaemon", "serve", "--host", "0.0.0.0", "--port", "8000"]
