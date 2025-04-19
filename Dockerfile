# Use the official Python 3.10 image from Docker Hub as a base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Set the working directory inside the container
WORKDIR $APP_HOME

# Copy the requirements.txt file first, then install dependencies
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port the app will run on
EXPOSE 8080

# Set the number of workers (adjust based on the load; typically, 2-4 workers per CPU core)
# Example: Using 4 workers here for handling multiple users
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8080", "app:app"]

