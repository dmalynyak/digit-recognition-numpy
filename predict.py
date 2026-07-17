import sys
import train
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_img(path, invert=False):
    img = Image.open(path).convert("L") # convert RGB to Greyscale (0-255) white-black
    img = img.resize((28, 28))
    arr = np.array(img)
    if invert: # mnist_784 from openml are pictures of white numbers on black background
        arr = 255 - arr
    arr = arr / 255.0
    return arr.reshape(784, 1)

def recognise(model, input_data):
    return model.predict(input_data)[0]

def main():

    path = sys.argv[1] # run in terminal:  python predict.py image_examples/file.png

    image = load_img(path) # white drawing on black background
    #image = load_img(path, invert=True) # black drawing on white background
    # check image:
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.show()


    model = train.NeuralNetwork([784, 128, 10])
    model.load_model("models/model_mini_b_784_128_10.pkl")

    print("prediction: ", recognise(model, image))

if __name__ == "__main__":
    main()