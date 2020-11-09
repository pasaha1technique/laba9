#Импрот
import tensorflow as tf
from tensorflow import keras
import numpy as np


#Загружаем датасет
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['Футболка / топ', "Шорты", "Свитер", "Платье",
              "Плащ", "Сандали", "Рубашка", "Кроссовок", "Сумка",
              "Ботинок"]

train_images = train_images / 255.0
test_images = test_images / 255.0

#Нейронная сеть
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Компилируем модель
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Тренируем модель
model.fit(train_images, train_labels, epochs=5)

#Сравниваем модель в тестовом наборе данных
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#Прогнозирование
predictions = model.predict(test_images)
print(predictions[0])

