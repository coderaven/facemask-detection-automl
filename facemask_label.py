import tensorflow as tf
import tensorflow_core.lite as tflite
import numpy as np
import PIL

MODEL_PATH = "facemask/model.tflite"
LABELS_PATH = "facemask/dict.txt"
IMAGE_PATH = "test_images/sample_1.jpg"


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def load_image_as_tensor_data(image_path, width, height):
    img = PIL.Image.open(image_path).resize((width, height))
    # add N dim
    return np.expand_dims(img, axis=0)


def classifyImage(image_path):
    labels = load_labels("facemask/dict.txt")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    
    # A tensor is a generalization of vectors and matrices and is easily understood as a multidimensional array.
    # For just think of tensors as "slots" or "containers" used by our AI Model to compute input and arrive to an output.
    # Eg. to predict a label.
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    interpreter.set_tensor(
        input_details[0]['index'], load_image_as_tensor_data(image_path, width, height))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(LABELS_PATH)

    for i in top_k:
        print(f'{float(results[i] / 255.0)}: {labels[i]}')

classifyImage(IMAGE_PATH)