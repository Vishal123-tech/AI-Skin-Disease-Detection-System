FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

EXPOSE 7860
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} app:app"]
