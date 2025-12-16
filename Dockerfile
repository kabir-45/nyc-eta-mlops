# 1. Use a lightweight Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the dependency list
COPY requirements.txt .

# 4. Install dependencies (no cache to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the source code folder
COPY src/ src/

# 6. Copy the trained model specifically
COPY models/production_model.pkl models/production_model.pkl

# 7. Expose the port the app runs on
EXPOSE 8000

# 8. Command to run the app
CMD ["uvicorn", "src.app.server:app", "--host", "0.0.0.0", "--port", "8000"]