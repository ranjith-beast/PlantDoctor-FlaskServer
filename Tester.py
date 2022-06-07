import tensorflow_core as tf
import json
import numpy as np
import cv2
from PIL import Image
with open("cat.json") as f:
  data = json.load(f)
# img=Image.open(filename)
model = tf.keras.models.load_model("Xception Model")
print(model.get_config())
image=cv2.imread("leaf blight.JPG")
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (98, 98))
img = img.reshape(1, 98, 98, 3)
img = img.astype('float32')
img=img/255.0
print(img)
predictions = model.predict(img)
class_names = np.argmax(predictions)
disease_name = data[str(class_names)]
print(disease_name)
print(class_names)