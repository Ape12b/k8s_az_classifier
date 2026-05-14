# Using Python 3.10 to better support newer scikit-learn 1.8.0 features
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
# Added libgomp1 which is often required for XGBoost in lean images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application and model artifacts
# Note: Ensure these .pkl files were exported using scikit-learn 1.8.0
COPY main.py .
COPY preprocessor.pkl .
COPY final_stack_f.pkl .
COPY final_stack_m.pkl .

# Expose FastAPI port
EXPOSE 80

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]