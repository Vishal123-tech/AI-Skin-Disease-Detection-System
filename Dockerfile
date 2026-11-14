FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV SKINSCANIX_MODEL=current

COPY requirements.txt .
COPY requirements/ ./requirements/
RUN pip install --no-cache-dir -r requirements/requirements-deploy.txt

COPY . .
RUN python scripts/assemble_dinov2.py

EXPOSE 7860
CMD ["sh", "-c", "python gradio_app.py"]
