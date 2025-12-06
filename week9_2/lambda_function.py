import json
import tensorflow as tf 
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import load_img
from keras.applications.xception import preprocess_input
from torchvision import transforms
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array

from prepare_image import download_and_prepare_image


def predict(url):
    # simple placeholder prediction - replace with real model logic
    return {"prediction": "pants"}


train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # ImageNet normalization
])

def preprocess_image(img):
    pass

# 1. Define the image path and target size
# Xception model generally uses a default input size of 299x299 pixels
# img_path = 'path/to/your/image.jpg'
#     target_size = (299, 299)

#     # 2. Load the image and resize it
#     img = load_img(img_path, target_size=target_size)

#     # 3. Convert the PIL image to a NumPy array
#     x = img_to_array(img)

#     # 4. Expand the dimensions to create a batch size of 1
#     # Shape changes from (299, 299, 3) to (1, 299, 299, 3)
#     x = np.expand_dims(x, axis=0)

#     # 5. Apply the Xception model's specific preprocessing function
#     # This scales the pixel values from [0, 255] to [-1, 1]
#     x = preprocess_input(x)


def predict_keras(url):

    model = keras.models.load_model('model_2024_hairstyle.keras')
    img = download_and_prepare_image(url, target_size=(200, 200))
    transform = train_transforms
    img = transform(img)
    print(img)


    # x = np.array(img)
    # print(x[0,0,0])
    # X = np.array([x])
    # print(X.shape)


    # X = preprocess_input(X)
    # print( "Question 3:", X[0,0,0,0])
    # print(X.shape)

    preds = ''# model.predict(X)
    #print(preds)

    return preds

def lambda_handler(event, context):
    print('parameters:', event)
    url = event['url']
    results = predict_keras(url)
    return results
