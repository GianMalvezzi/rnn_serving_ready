FROM tensorflow/tfx:1.0.0

# Copy project files
COPY . /app
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app