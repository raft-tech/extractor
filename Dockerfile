FROM python:3.11.0a5

COPY . /scene_text_extractor
WORKDIR /scene_text_extractor
RUN pip install -r requirements.txt

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "wsgi:app", "--timeout", "100", "--graceful-timeout", "50", "--max-requests-jitter", "2000", "--max-requests", "50", "-w", "2", "--keep-alive", "2"]