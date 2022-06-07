import tensorflow as tf
model = tf.keras.models.load_model("firstmodel")
print(model.summary())