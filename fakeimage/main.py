import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from scipy.fft import dct
from google.cloud import vision
import io

# Load the trained models
model_human_classifier = load_model('human_classifier_model_with_vgg16.h5')
model_fake_detector = load_model('fake_image_classifier_model.h5')



def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array /= 255.0  # Rescale pixel values
    return img_array


# Function to classify the image as human or non-human
def classify_image(img_array, model):
    prediction = model_human_classifier.predict(img_array)
    return "Human" if prediction >= 0 else "Non-human"

# Function to check if the human image is real
def verify_human_image(img_array, fake_detector_model):
    prediction = fake_detector_model.predict(img_array)
    return "Real" if prediction >= 0.5 else "Fake"

# Function to check if the non-human image is edited
def detect_edits_in_non_human_image(img_array):
    dct_coeffs = dct(dct(np.array(img_array[0], dtype=np.float32), axis=0), axis=1)
    high_freq_energy = np.sum(dct_coeffs[8:, 8:]) / np.sum(dct_coeffs)
    threshold = min(20, 100 * high_freq_energy)
    return "Edited" if high_freq_energy > threshold else "Not Edited"

# Example usage:
image_path = 'realorfake/fake/easy_5_1100.jpg'
img_array = preprocess_image(image_path)

# Classify the image as human or non-human
classification_result = classify_image(img_array, model_human_classifier)

# Verify if the human image is real or fake
if classification_result == "Human":
    verification_result = verify_human_image(img_array, model_fake_detector)
else:
    verification_result = detect_edits_in_non_human_image(img_array)

print("Classification Result:", classification_result)
print("Verification Result:", verification_result)


# Initialize the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

def generate_image_description(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Create an image object
    image = vision.Image(content=content)

    # Perform label detection on the image
    response = client.annotate_image({
      'image': image,
      'features': [{'type': vision.Feature.Type.IMAGE_PROPERTIES}, 
                   {'type': vision.Feature.Type.LABEL_DETECTION},
                   {'type': vision.Feature.Type.OBJECT_LOCALIZATION},
                   {'type': vision.Feature.Type.LOGO_DETECTION},
                   {'type': vision.Feature.Type.TEXT_DETECTION},
                   {'type': vision.Feature.Type.DOCUMENT_TEXT_DETECTION},
                   {'type': vision.Feature.Type.WEB_DETECTION},
                   {'type': vision.Feature.Type.FACE_DETECTION},
                   {'type': vision.Feature.Type.IMAGE_CLASSIFICATION},
                   {'type': vision.Feature.Type.IMAGE_CONTEXT},
                   {'type': vision.Feature.Type.CROP_HINTS},
                   {'type': vision.Feature.Type.SAFE_SEARCH_DETECTION},
                   {'type': vision.Feature.Type.LANDMARK_DETECTION},
                   {'type': vision.Feature.Type.FACE_DETECTION},
                   {'type': vision.Feature.Type.SHOT_CHANGE_DETECTION},
                   {'type': vision.Feature.Type.COLOR_INFO}
                   ]
    })

    # Extract the description from the response
    description = response.text_annotations[0].description

    return description


# Example usage:
image_path = 'sample.jpeg'
description = generate_image_description(image_path)
print("Image description:", description)