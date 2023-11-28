FROM python:3.9-slim-buster
WORKDIR /app
COPY . /app

# Install AWS CLI
RUN apt-get update -y && apt-get install -y awscli

# Install system dependencies
RUN apt-get install -y ffmpeg libsm6 libxext6 unzip

# Install Python dependencies
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]