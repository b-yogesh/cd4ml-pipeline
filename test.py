import tensorflow as tf
from tensorflow import keras
import dvc.api
import numpy as np


def main():
    model = keras.models.load_model("cnn_model")
    
    ## using dvc to access data
    path = 'data/mnist.npz'
    repo= 'D:/cd4ml/cd4ml-pipeline'
    version='v1'

    data_url = dvc.api.get_url(
        path=path,
        repo=repo,
        rev=version
    )

    with np.load(data_url, allow_pickle=True) as g:
        a = g['arr_0'].item()
        with np.load(a) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']

    x_test = x_test[..., tf.newaxis].astype("float32")[:2000]
    y_test = y_test[:2000]
    predictions = model(x_test, training=False)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    test_accuracy(y_test, predictions)
    print('Test Accuracy ::', float(test_accuracy.result()))


if __name__ == '__main__':
    main()

