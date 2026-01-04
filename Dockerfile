# Default Dockerfile - points to Pathway pipeline
# For Railway deployment, see Dockerfile.railway
# For local docker-compose, this uses the Pathway pipeline

FROM pathwaycom/pathway:latest

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install -U --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose webhook port for message ingestion
EXPOSE 8080

CMD ["python", "app.py"]
