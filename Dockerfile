# Research Agent — HuggingFace Spaces (Docker SDK) imajı
# Yerelde:  docker build -t research-agent . && docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... research-agent
FROM python:3.11-slim

# HF Spaces'in önerdiği pratik: root olmayan kullanıcı (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Önce bağımlılıklar (katman önbelleği). Not: bu projede torch yok → imaj hafif.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Uygulama kodu
COPY --chown=user . /app

# HuggingFace Spaces 7860 portunu bekler
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
