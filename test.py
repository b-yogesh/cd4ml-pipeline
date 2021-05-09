import tensorflow as tf
from tensorflow import keras


def main():
    model = keras.models.load_model("cnn_model")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test[..., tf.newaxis].astype("float32")[:2000]
    y_test = y_test[:2000]
    predictions = model(x_test, training=False)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    test_accuracy(y_test, predictions)
    print('Test Accuracy ::', test_accuracy.non_trainable_variables[0]/test_accuracy.non_trainable_variables[1])


if __name__ == '__main__':
    main()

