# Menggunakan image Python 3.10-slim
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies jika diperlukan
# RUN apt-get update && apt-get install -y build-essential

# Copy requirements.txt terlebih dahulu untuk memanfaatkan cache Docker
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy seluruh proyek ke dalam Docker image
COPY . .

# Expose port yang digunakan oleh aplikasi
EXPOSE 5000

# Jalankan aplikasi dengan Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.main:app"]

