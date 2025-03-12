# Use the official Python 3.11 slim image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
ADD . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt





# Run the Flask application
CMD ["python", "app.py"]
