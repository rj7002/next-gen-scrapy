# NGS Explorer on a Hugging Face Docker Space: build the React app, then run the FastAPI server on port 7860.

FROM node:24-slim AS web
WORKDIR /web
COPY service/web/package.json service/web/package-lock.json ./
RUN npm ci
COPY service/web/ ./
RUN npm run build

FROM python:3.14-slim
# OpenCV's headless wheel still wants these at import time
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user pyproject.toml LICENSE ./
COPY --chown=user next_gen_scrapy ./next_gen_scrapy
RUN pip install --no-cache-dir --user --no-deps .
COPY --chown=user service ./service
COPY --chown=user --from=web /web/dist ./service/web/dist
EXPOSE 7860
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "7860"]
