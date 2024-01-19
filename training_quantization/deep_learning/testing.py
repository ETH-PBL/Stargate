import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="../quantization_deployment_gap_sdk/tflite_models/gate_navigator_model_v22.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

tensor_details = interpreter.get_tensor_details()
image_details = interpreter.get_input_details()[0]
tof_details = interpreter.get_input_details()[1]
output_details = interpreter.get_output_details()[0]
print('image: ', image_details["name"])
print('tof: ', tof_details["name"])
input_scale, input_zero_point = image_details["quantization"]
print("Image scale/zero point: ", input_scale, input_zero_point)
input_scale, input_zero_point = tof_details["quantization"]
print("tof scale/zero point: ", input_scale, input_zero_point)
output_scale, output_zero_point = output_details["quantization"]
print("output scale/zero point: ", output_scale, output_zero_point)
print("\n\n\n\n")
for dict in tensor_details:
    #print(dict)
    i = dict['index']
    tensor_name = dict['name']
    scales = dict['quantization_parameters']['scales']
    zero_points = dict['quantization_parameters']['zero_points']
    try:
        tensor = interpreter.tensor(i)()

        print(i, tensor_name, ",Shape scales: ", scales.shape,",Input Shape: ", tensor.shape, ",Zero_point shape and value: ", zero_points.shape, zero_points)
    except:
        print()