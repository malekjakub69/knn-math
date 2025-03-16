FROM python:3.11-slim

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY app/ .  

# CMD ["python", "main.py"]
ENTRYPOINT [ "python", "-m", "main" ]
CMD ["--train"]     # Default flag, can be overridden