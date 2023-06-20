from flask import Flask, request, make_response
import json
import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

data = pd.read_csv("food_dataset.csv")
model = load_model('calories_est.h5')


food_arr = []
image_src = []
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route('/')
def home():
    return 'Welcome to Food Calories Estimation'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        # Get the image file from the POST request
        image_file = request.files['image']
        if image_file.filename != '':

            # Save the uploaded image to a specified location
            image_location = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_location)

            # Read and preprocess the image using OpenCV and TensorFlow
            cv2_img = cv2.cvtColor(cv2.imread(image_location), cv2.COLOR_BGR2RGB)
            resize = tf.image.resize(cv2_img, (224, 224))

            # Perform the prediction using the loaded model
            prediction = model.predict(np.expand_dims(resize / 255, axis=0))
            max_prediction = prediction[0].max()

            # Retrieve food categories from the dataset and sort them alphabetically
            food_category = data['Food'].unique()
            food_category.sort()

            # Store the image location and clear the food_arr list
            image_src.append(image_location)
            food_arr.clear()

            # Find the food category with the maximum prediction value
            # and retrieve the food's details from the dataset using the index of the food category
            for i in range(len(prediction[0])):
                if prediction[0][i] == max_prediction:
                    food_arr.append(food_category[i])
                    idx = data.index[data['Food'] == food_category[i]].tolist()
                    food_arr.append(data['Category'][idx[0]])
                    food_arr.append(data['Calories'][idx[0]])
                    food_arr.append(data['Protein'][idx[0]])
                    food_arr.append(data['Fat'][idx[0]])
                    food_arr.append(data['Sat.Fat'][idx[0]])
                    food_arr.append(data['Fiber'][idx[0]])
                    food_arr.append(data['Carbs'][idx[0]])
                    break

            # Create a response JSON with the predicted food information
            api_data = {
                'message': 'Success Prediction',
                'food_name': food_arr[0],
                'category': food_arr[1],
                'calories': food_arr[2],
                'protein': food_arr[3],
                'fat': food_arr[4],
                'sat_fat': food_arr[5],
                'fiber': food_arr[6],
                'carbs': food_arr[7],
                'image_src': image_src[0]
            }

            # Return the response JSON with a success status code (200)
            return make_response(json.dumps(api_data, cls=NpEncoder), 200)


if __name__ == '__main__':
    app.run()
