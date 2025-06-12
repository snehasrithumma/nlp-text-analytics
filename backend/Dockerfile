FROM python:3.12-slim

WORKDIR /usr/src/app

COPY app ./app
COPY requirements.txt .
COPY logs ./logs

RUN pip install --no-cache-dir -r requirements.txt
RUN chmod -R 777 ./logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]