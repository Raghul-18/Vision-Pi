import keras.applications
import numpy as np
from keras_preprocessing import image
from keras.applications import imagenet_utils
import pyttsx3


def tts(n):
    engine = pyttsx3.init()
    engine.say(n)
    engine.runAndWait()


filename = "C:/Users/RAGHUL/Downloads/bicycle_original.jpeg"
objs = []
mobile = keras.applications.mobilenet_v2.MobileNetV2()
img = image.load_img(filename, target_size=(224, 224))
resized_image = image.img_to_array(img)
final_img = np.expand_dims(resized_image, axis=0)
final_img = keras.applications.mobilenet_v2.preprocess_input(final_img)
predictions = mobile.predict(final_img)
result = imagenet_utils.decode_predictions(predictions, top=5)
res = np.array(result)
res = res.flatten()
objs.append(res[1])
objs.append(res[4])
print(objs)
tts(objs)
