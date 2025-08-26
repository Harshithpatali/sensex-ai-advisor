# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8503

# Run Streamlit
CMD ["streamlit", "run", "app/dashboard.py", "--server.port", "8501", "--server.address=0.0.0.0"]
