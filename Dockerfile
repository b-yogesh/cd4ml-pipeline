FROM tensorflow/tensorflow:2.5.0rc3

COPY . /app

WORKDIR /app

CMD python src/train.py