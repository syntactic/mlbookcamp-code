FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN apt-get update
RUN apt-get -y install gcc

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

COPY ["train.py", "predict.py", "ridge_regressor_polynomial_alpha_1e-1.bin", "spotify_dict_vectorizer.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
