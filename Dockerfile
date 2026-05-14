# Use a slim Python image to keep the footprint small for your laptops
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (required for some scikit-learn/pandas operations)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the model files and the API script into the container
COPY main.py .
COPY preprocessor.pkl .
COPY final_stack_f.pkl .
COPY final_stack_m.pkl .

# Expose the port FastAPI will run on
EXPOSE 80

# Command to run the API using Uvicorn
# We use 0.0.0.0 so it's accessible outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]