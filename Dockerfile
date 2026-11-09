# Use an official lightweight Python image (same version as my virtual environment)
FROM python:3.14-slim

# Set the working directory inside the container
WORKDIR /app

# Install packages first, so this step is cached when only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code into the container
COPY src/ ./src/

# Run the script
CMD ["python", "src/app/main.py"]

