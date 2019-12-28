import tensorflow as tf
from tensorflow.keras.datasets.fashion_mnist import load_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


(X_train, y_train), (X_test, y_test) = load_data()


X_train.shape

y_train.shape

X_train[0]


plt.figure()
plt.imshow(X_train[0], cmap='Greys')
plt.colorbar()
plt.grid(False)
plt.show()

y_train[0]


plt.imshow(X_train[10], cmap='Greys')
plt.colorbar()
plt.grid(False)
plt.show()

y_train[10]

mapa_kategorii = {
    0 : 'T-shirt/top',
    1 : 'Trouser',
    2 : 'Pullover',
    3 : 'Dress',
    4 : 'Coat',
    5 : 'Sandal',
    6 : 'Shirt',
    7 : 'Sneaker',
    8 : 'Bag',
    9 : 'Ankle boot'
}


mapa_kategorii[y_train[10]]

X_train = X_train / 255.0

y_train = y_train.astype(np.int32)

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=2)


loss, accuracy = model.evaluate(X_train, y_train, verbose=2)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

X_test = X_test / 255.0

y_test = y_test.astype(np.int32)

y_pred = model.predict(X_test, steps = 10)
print(y_pred.shape)

y_pred[0]

np.argmax(y_pred[0])

mapa_kategorii[np.argmax(y_pred[0])]

mapa_kategorii[y_test[0]]

plt.imshow(X_test[0], cmap='Greys')
plt.colorbar()
plt.grid(False)
plt.show()

plt.grid(False)
plt.xticks(range(10))
plt.yticks([])
plt.bar(range(10), y_pred[0])
plt.ylim([0, 1])

loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print("Loss :", loss)
print("Accuracy :", accuracy)


nr_kategorii = 0
mapa_aktywacji = []
for elem in model.weights[0].numpy():
    mapa_aktywacji.append(elem[nr_kategorii])
    
plt.imshow(np.array(mapa_aktywacji).reshape(28, 28))


def decode_img(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [28, 28])

lista_plikow = tf.data.Dataset.list_files(str('/home/kba/Pobrane/buty.jpg'))

moj_test = lista_plikow.map(decode_img)

obraz = None
for elem in moj_test.take(1):
    obraz = elem.numpy()

plt.imshow(obraz.reshape(28, 28), cmap='Greys')
plt.show()

y_pred_moj = model.predict(obraz.reshape(1, 784))

y_pred_moj

mapa_kategorii[np.argmax(y_pred_moj)]


plt.grid(False)
plt.xticks(range(10))
plt.yticks([])
plt.bar(range(10), y_pred_moj[0])
plt.ylim([0, 1])

model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model2.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
    metrics=['accuracy']
)

model2.fit(X_train.reshape(60000, 28, 28, 1), y_train, epochs=2)

for weight in model2.weights:
    print(weight.shape)
    
    