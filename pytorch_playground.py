from deepface import DeepFace
from deepface.modules import verification
import numpy as np

from deepface.models.FacialRecognition import FacialRecognition
import cv2

import ezkl
import os
import json

import tf2onnx
import onnx

import torch
from onnx2torch import convert


model_name = "OpenFace"
model: FacialRecognition = DeepFace.build_model(model_name)

onnx_model, _ = tf2onnx.convert.from_keras(model.model)
onnx.save(onnx_model, "openface_new.onnx")
pytorch_model = convert(onnx_model)

pytorch_model.eval()

target_size = model.input_shape

img1 = DeepFace.extract_faces(img_path="emmapassport.png")[0]["face"]
img1 = cv2.resize(img1, target_size)
img1 = np.expand_dims(img1, axis=0)  # to (1, 224, 224, 3)

img1_torch = torch.tensor(img1, dtype=torch.float32)

img1_representation = pytorch_model.forward(img1_torch)

img1_representation = np.array(img1_representation)
# Export the model
# torch.onnx.export(pytorch_model,               # model being run
#                       img1_torch,                   # model input (or a tuple for multiple inputs)
#                       "openface_pt.onnx",            # where to save the model (can be a file or file-like object)
#                       export_params=True,        # store the trained parameter weights inside the model file
#                       opset_version=10,          # the ONNX version to export the model to
#                       do_constant_folding=True,  # whether to execute constant folding for optimization
#                       input_names = ['input'],   # the model's input names
#                       output_names = ['output'], # the model's output names
#                       dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                                     'output' : {0 : 'batch_size'}})

# data_array = (img1).reshape([-1]).tolist()
# data = dict(input_data = [data_array])
# data_path = os.path.join('input.json')
# settings_path = os.path.join('settings.json')
# model_path = os.path.join('openface_pt.onnx')

# # Serialize data into file:
# json.dump( data, open(data_path, 'w' ))

# py_run_args = ezkl.PyRunArgs()
# py_run_args.input_visibility = "private"
# py_run_args.output_visibility = "public"
# py_run_args.param_visibility = "fixed" # private by default

# # What is the settings file?
# res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)

# assert res == True

# cal_path = os.path.join("calibration.json")

# img1 = DeepFace.extract_faces(img_path="emmapassport2.jpg")[0]["face"]
# img1 = cv2.resize(img1, target_size)
# img1 = np.expand_dims(img1, axis=0)  # to (1, 224, 224, 3)

# data_array = (img1).reshape([-1]).tolist()
# data = dict(input_data = [data_array])

# json.dump(data, open(cal_path, 'w'))

# ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")