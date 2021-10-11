FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN ["pip", "install", "pipenv"]

WORKDIR app

COPY ["Pipfile", "Pipfile.lock", "server.py", "dv.bin", "model1.bin","."]
RUN ["pipenv", "install", "--system", "--deploy"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "server:app"]
