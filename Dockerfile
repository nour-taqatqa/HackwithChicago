FROM pathwaycom/pathway:latest

WORKDIR /app

RUN apt-get update \
    && apt-get install -y python3-opencv tesseract-ocr-eng poppler-utils \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY requirements.txt .
RUN pip install -U --no-cache-dir -r requirements.txt

RUN pip install -U "pathway[all]"

COPY . .

CMD ["python", "app.py"]