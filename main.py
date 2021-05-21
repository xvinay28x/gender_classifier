from tensorflow import keras
train = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_set = train.flow_from_directory("image/",
                                     target_size=(255,255),
                                     batch_size=5,
                                     subset="training",
                                     class_mode="binary")

model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=16, kernel_size=3, input_shape=(255,255,3), activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dense(units=1, activation="sigmoid"))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_set,epochs=10)

model.save("gender_classifier.h5")