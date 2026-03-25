FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY rag_system.py .
COPY rag_filters.py .
COPY date_normalizer.py .
COPY adaptive_chunker.py .
COPY semantic_chunker.py .

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
