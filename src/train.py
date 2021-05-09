import tensorflow as tf
from model.cnn import ConvNet
from tensorflow.python.keras.utils.data_utils import get_file
import numpy as np
import mlflow
import mlflow.tensorflow
import dvc.api

EPOCHS = 2
BATCH_SIZE = 32

# code from tensorflow - just to save mnist model to use dvc
# origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
# path='mnist.npz'
# path = get_file(
#         path,
#         origin=origin_folder + 'mnist.npz',
#         file_hash=
#         '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')

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
# np.savez('data/mnist.npz', path)

## mlflow init
mlflow.set_experiment('cnn_model_exp')

# normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# add new dim for channel
x_train = x_train[..., tf.newaxis].astype("float32")[:2000]
y_train = y_train[:2000]
x_test = x_test[..., tf.newaxis].astype("float32")[:200]
y_test = y_test[:200]

# split data in train/test
train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

print('Data loaded')
# model
model = ConvNet()

# loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# optimizer
optimizer = tf.keras.optimizers.Adam()

# metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

mlflow.log_param('data_url', data_url)
mlflow.log_metric('test_loss', float(test_loss.result()))
mlflow.log_metric('test_accuracy', float(test_accuracy.result()))

def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def train():
    print('Starting training')
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
        model.save("cnn_model")
        print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
        )
    print('Training Completed')

if __name__ == '__main__':
    train()

# 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