FROM python:3.9.16-slim-buster
COPY . .
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python", "-u"]
CMD ["./src/main.py"]