import keras
import tensorflow as tf
import matplotlib as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

xTrain = tf.keras.utils.normalize(xTrain, axis=1)
xTest = tf.keras.utils.normalize(xTest, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # Used to be 128 on both
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(xTrain, yTrain, epochs=1000)

val_loss, val_acc = model.evaluate(xTest, yTest)
print(val_loss, val_acc)

model.save('defaultClassifier')

#model = tf.keras.models.load_model('defaultClassifier')
#predictions = model.predict([xTest])
#print(xTest[0])

#print(np.argmax(predictions[0]))

# print(xTrain)
# plt.imshow(xTrain, cmap=plt.cm.binary)
# plt.show()
