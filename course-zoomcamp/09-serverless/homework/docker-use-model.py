import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
#from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy

model_path = "cats-dogs-v2.tflite"
image_path = "https://upload.wikimedia.org/wikipedia/commons/1/18/Vombatus_ursinus_-Maria_Island_National_Park.jpg"

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

print(input_index, output_index)

def preprocess_input(x):
    x /= 255.
    return x

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def lambda_handler(event, context):
    image_path = event['url']
    img = download_image(image_path)
    img = prepare_image(img, (150,150))
    x = numpy.asarray(img, dtype=numpy.float32)
    #x = image.img_to_array(img)
    x = preprocess_input(x)

    print(x[0][0])

    interpreter.set_tensor(input_index, [x])
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    print(preds)
    return preds[0].tolist()
