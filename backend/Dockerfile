# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app/backend

# Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# ENV PYTHONPATH="/app/backend"

# Copy the application code
COPY backend/ .

# Expose the FastAPI port
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
