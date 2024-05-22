from deepface import DeepFace
from deepface.modules import verification
import numpy as np

from deepface.models.FacialRecognition import FacialRecognition
import cv2

import ezkl
import os
import json

model_name = "OpenFace"
model: FacialRecognition = DeepFace.build_model(model_name)

# model_to_save = model.model
# model_to_save.save("openface")

target_size = model.input_shape

img1 = DeepFace.extract_faces(img_path="emmapassport.png")[0]["face"]
img1 = cv2.resize(img1, target_size)
img1 = np.expand_dims(img1, axis=0)  # to (1, 224, 224, 3)

img1_representation = model.forward(img1)

img1_representation = np.array(img1_representation)

print(img1.shape)

data_array = (img1).reshape([-1]).tolist()

data = dict(input_data = [data_array])
print(len(data_array))

data_path = os.path.join('input.json')
settings_path = os.path.join('settings.json')
model_path = os.path.join('openface.onnx')

# Serialize data into file:
json.dump( data, open(data_path, 'w' ))

py_run_args = ezkl.PyRunArgs()
py_run_args.input_visibility = "private"
py_run_args.output_visibility = "public"
py_run_args.param_visibility = "fixed" # private by default

# What is the settings file?
res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)

assert res == True
