# Use a modern slim Python base
FROM python:3.11-slim

# Noninteractive apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/.matplotlib
ENV PORT=3000

# Create app directory
WORKDIR /app

# Install system-level deps needed for pyodbc + msodbcsql + building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    wget \
    build-essential \
    unixodbc \
    unixodbc-dev \
    locales \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft package signing key and repo (safer keyring method)
RUN set -eux; \
    mkdir -p /usr/share/keyrings; \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
      > /etc/apt/sources.list.d/mssql-release.list; \
    apt-get update; \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18; \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first (cache-friendly)
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python deps (including gunicorn)
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt && \
    pip install gunicorn

# Copy application code last to maximize layer cache
COPY . /app


# Create a non-root user (optional but recommended)
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 3000

# Run command
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:3000", "--workers", "2", "--threads", "2"]
