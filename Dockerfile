FROM python:3.8-slim-buster

RUN apt get update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]