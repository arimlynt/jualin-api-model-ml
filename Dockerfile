# Base image
FROM tensorflow/tensorflow:latest

# Copy source code to container
COPY . /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install -r /app/requirements.txt

# Set working directory
WORKDIR /app

# Expose port 5000 for Flask app
EXPOSE 5000

# Start Flask app
CMD ["python3", "main.py"]