import numpy as np
import tensorflow as tf

X = np.random.randint(low=1, high=20, size=1000)
y = 2*X + 1

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X, y, epochs=100)

print(f'Actual: f(9) -> 19\nPredicted: f(9) -> {model.predict([9])}')

model.save('sample.h5')