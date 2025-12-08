FROM python:3.11

# Set working directory inside the container.
WORKDIR /app

# Install dependencies.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r ./requirements.txt

# Copy files from virtual machine to Docker container.
COPY src/ ./src/
COPY data/ ./data/
COPY models/voting_model.joblib ./models/voting_model.joblib

# Start Flask web server.
CMD ["python", "-m", "src.api.server"]