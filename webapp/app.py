from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from tensorflow.keras.datasets import mnist

app = Flask(__name__)

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Function to get images of a specific digit
def get_images(digit, count):
    images = []
    digit = int(digit)
    idxs = np.where(y_train == digit)[0][:count]
    for idx in idxs:
        img = x_train[idx]
        buf = BytesIO()
        plt.imsave(buf, img, cmap='gray')
        data = base64.b64encode(buf.getvalue()).decode('utf-8')
        images.append({
            'digit': digit,
            'data': data
        })
    return images

@app.route('/', methods=['GET', 'POST'])
def index():
    images = []
    if request.method == 'POST':
        digit = request.form['digit']
        count = int(request.form['count'])
        images = get_images(digit, count)
    return render_template('index.html', images=images)

if __name__ == '__main__':
    app.run(debug=True)
