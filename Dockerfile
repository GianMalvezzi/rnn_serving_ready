FROM tensorflow/tfx:1.0.0

# Copy project files
COPY . /app
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app

#install pandas and create the requirements file
RUN pip install pandas
RUN pip freeze > requirements.txt