# Dockerfile for Hugging Face Spaces (Docker SDK) running the Streamlit app.
FROM python:3.11-slim

# Hugging Face Spaces runs the container as a non-root user (uid 1000).
# Give that user a writable home so model/cache downloads succeed.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# Install CPU-only PyTorch first so requirements.txt doesn't pull the large
# CUDA build (smaller image, faster build) on this CPU-only Space.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# HF Spaces expects the app on the port declared as app_port in README (7860).
EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
