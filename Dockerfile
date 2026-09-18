FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

COPY requirements.txt requirements-deploy.txt .
RUN pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

EXPOSE 7860
CMD ["sh", "-c", "python gradio_app.py"]
