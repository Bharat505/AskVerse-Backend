# Use official Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . .

# Copy the service account key into the container
COPY vertex-ai-key.json /app/vertex-ai-key.json

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the correct port for Cloud Run
EXPOSE 8080

# Set the environment variable for Google authentication
# (Cloud Run allows setting this as a Secret Environment Variable)
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/vertex-ai-key.json

# Run the application on Cloud Run's expected port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
