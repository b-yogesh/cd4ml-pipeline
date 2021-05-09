import tensorflow as tf
from model.cnn import ConvNet

EPOCHS = 2
BATCH_SIZE = 32

mnist = tf.keras.datasets.mnist
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# get mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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