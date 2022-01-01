from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model = load_model('image_classification_model.h5')

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(150,150))
    i = image.img_to_array(i)
    i = np.expand_dims(i, axis=0)
    i = np.vstack([i])
    p = model.predict(i, batch_size=10)
    return p[0]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about", methods=['GET', 'POST'])
def about_page():
    return "Created by Hema..!!!"

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
    
        img_path = "static/" + img.filename    
        img.save(img_path)
        p = predict_label(img_path)
        label = ""
        if p > 0:
            label = "Dog"
        else:
            label = "Cat"
        return render_template("index.html", prediction = label, img_path = img_path)

if __name__ =='__main__':
    app.run()