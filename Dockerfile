# Use full Python image so matplotlib & duckdb work without extra OS deps
FROM python:3.11

# Workdir inside the container
WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py utils.py index.html ./

# If you have these folders, copy them too.
# (If you don't have them yet, either create empty folders or delete these two lines.)
COPY static ./static
COPY data ./data

# Make logs unbuffered
ENV PYTHONUNBUFFERED=1

# Render will inject $PORT; your app already uses os.getenv("PORT", "3000") :contentReference[oaicite:1]{index=1}
EXPOSE 3000

# Use gunicorn to serve the Flask app: "server:app"
CMD gunicorn -b 0.0.0.0:$PORT server:app
