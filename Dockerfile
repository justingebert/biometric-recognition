FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/backend/requirements.txt app/backend/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r app/backend/requirements.txt

COPY app/frontend/package*.json app/frontend/
RUN npm ci --prefix app/frontend

COPY app app

ENV PYTHON=/opt/venv/bin/python
EXPOSE 5173 8000

CMD ["./app/start.sh"]
