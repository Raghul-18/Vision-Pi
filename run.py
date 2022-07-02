from Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
classFile = "coco.names"
imagePath = "C:/Users/RAGHUL/Downloads/bicycle_original.jpeg"

detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImage(imagePath)
