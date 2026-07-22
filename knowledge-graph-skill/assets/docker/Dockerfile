FROM python:3.11-slim

WORKDIR /app

COPY assets/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY assets/ ./assets/
COPY SKILL.md .

EXPOSE 8700

CMD ["python", "scripts/kg_server.py", "--host", "0.0.0.0", "--port", "8700"]
