FROM python:3.11-slim

# 1. Force Python to print logs immediately so we never get a blank screen again!
ENV PYTHONUNBUFFERED=1

# 2. Create the specific non-root user required by Hugging Face (User ID 1000)
RUN useradd -m -u 1000 user
USER user

# 3. Set the environment variables for the new user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 4. Tell Hugging Face to save downloaded AI models to a folder this user owns
ENV HF_HOME=$HOME/app/.cache

# 5. Set the working directory
WORKDIR $HOME/app

# 6. Copy requirements and install them securely
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your application code
COPY --chown=user . .

EXPOSE 7860

# 8. Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]