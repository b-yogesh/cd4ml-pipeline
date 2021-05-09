
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class ConvNet(Model):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.conv2 = Conv2D(64, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

if __name__ == '__main__':
    cnn = ConvNet()
    input_shape = (1, 20, 20, 3)
    x = tf.random.normal(input_shape)
    y = cnn(x)
    print(y.shape) # (1, 10)