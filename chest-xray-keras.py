# from keras.datasets.fashion_mnist import load_data
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import numpy as np
import cv2
import os
import logging
import load_data_entry as db

#target_size
_target_size=96

#images folderpath
images_folder = os.path.join('..', 'ChestXray-NIHCC', 'images')

#database
df = db.data_entries()
x_data, y_data = db.load_data(df)

def load_image(img_path, target_size=(_target_size, _target_size), images_folder=images_folder):
  img = load_img(os.path.join(images_folder, img_path), target_size=target_size)
  return img_to_array(img)

def database(x_samples, y_samples, batch_size=100):
  num_batches = len(x_samples) // batch_size
  x_batches = np.array_split(x_samples, num_batches)
  y_batches = np.array_split(y_samples, num_batches)

  for b in range(len(x_batches)):
    x = np.array(list(map(load_image, x_batches[b])))
    y = np.array(y_batches[b])

    yield x, y

def preprocessing():
  return ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=False,
    vertical_flip=False
  )

def build_model():
  input_tensor = Input(shape=(_target_size, _target_size, 3))
  base_model = MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_tensor=input_tensor,
    input_shape=(_target_size, _target_size, 3),
    pooling='avg'
  )

  for layer in base_model.layers:
    layer.treinable = True # trainable has to be false in order to freeze layers

  op = Dense(256, activation='relu')(base_model.output)
  op = Dropout(.25)(op)

  # softmax layer
  output_tensor = Dense(15, activation='sigmoid')(op)

  model = Model(inputs=input_tensor, outputs=output_tensor)
  return model

# for x_train, y_train in database(x_data, y_data):
#   print('oi')
#   input()

model = build_model()
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

with open('model.json', 'w') as json_model:
  json_model.write(model.to_json())

logging.basicConfig(filename='metrics.log', level=logging.DEBUG)

n_epoch = 1000
loss = 0
for e in range(n_epoch):
  print('epoch', e)
  for x_train, y_train in database(x_data, y_data, batch_size=100):
    for x_batch, y_batch in preprocessing().flow(x_train, y_train, batch_size=32):
      metrics = model.train_on_batch(x_batch,y_batch)
      logging.debug('%s,%s' % (model.metrics_names, metrics))
  model.save_weights('model_weights.h5')
  print('Weights saved.')

# model.fit_generator(database(x_data, y_data), steps_per_epoch=900, verbose=1, epochs=5)