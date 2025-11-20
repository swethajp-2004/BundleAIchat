# Use a full Python image so matplotlib + duckdb Just Work™
FROM python:3.11

# Runtime envs
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/.matplotlib

# App directory
WORKDIR /app

# Install Python deps first (better layer cache)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy the rest of the application
# (server.py, utils.py, index.html, static/, data/, etc.)
COPY . /app

# Expose app port (Render will set $PORT, but 3000 is our internal default)
EXPOSE 3000

# Gunicorn entrypoint: use $PORT if set (Render), else 3000 (local)
CMD ["sh", "-c", "gunicorn server:app --bind 0.0.0.0:${PORT:-3000} --workers 2 --threads 2"]
