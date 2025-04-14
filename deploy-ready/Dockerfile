# Use the official Python image
FROM python:3.10-slim

# Set environment vars
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Cloud Run's required port
EXPOSE 8080

# Run Streamlit on the required port
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]
