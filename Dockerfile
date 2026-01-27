FROM python:3.11-slim

WORKDIR /app

# Faster, cleaner logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install your package (src layout)
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir -e .

# Copy API app
COPY api ./api

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
