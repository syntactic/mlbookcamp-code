import tensorflow.lite as tflite
from io import BytesIO
from urllib import request
from tensorflow.keras.preprocessing import image
from PIL import Image

model_path = "/Users/syntactic/Downloads/dogs_cats_10_0.687.tflite"

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

img = download_image("https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg")
img = prepare_image(img, (150,150))
x = image.img_to_array(img)
x = preprocess_input(x)

print(x[0][0])

interpreter.set_tensor(input_index, [x])
interpreter.invoke()
preds = interpreter.get_tensor(output_index)
print(preds)
