from google.cloud import vision
from google.cloud.vision_v1 import types

import io
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS']='vision-414807-5c7c9e0314a6.json'

# Initialize the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

def classify_image(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Create an image object
    image = types.Image(content=content)

    # Perform label detection on the image
    response = client.label_detection(image=image)
    labels = response.label_annotations

    # Extract label descriptions
    label_descriptions = [label.description.lower() for label in labels]

    # Determine if the image is human or not
    human_presence = "Human" if "person" in label_descriptions else "Not Human"

    if human_presence == "Human":
        # Perform face detection
        face_response = client.face_detection(image=image)
        faces = face_response.face_annotations
        if faces:
            # If faces are detected, classify if it's real or fake
            classification = "Real"  # Dummy implementation
            comment = "This image contains a person."
        else:
            classification = "Fake"  # Dummy implementation
            comment = "This image contains a person but it appears to be fake."
    else:
        # If not human, check if the image is edited or not
        safe_search_response = client.safe_search_detection(image=image)
        safe_search_annotations = safe_search_response.safe_search_annotation
        violence_likelihood = safe_search_annotations.violence
        is_edited = violence_likelihood in [types.Likelihood.LIKELY, types.Likelihood.VERY_LIKELY]
        if is_edited:
            classification = "Edited"
            comment = "This image does not contain a person but it appears to be edited."
        else:
            classification = "Not Edited"
            comment = "This image does not contain a person and it does not appear to be edited."

    # Perform image description
    response = client.image_properties(image=image)
    props = response.image_properties_annotation
    description = ""
    for prop in props.dominant_colors.colors:
        color = "(R:{}, G:{}, B:{})".format(round(prop.color.red), round(prop.color.green), round(prop.color.blue))
        description += str(round(prop.pixel_fraction * 100, 2)) + "% " + color + ", "

    return human_presence, classification, comment, description

def detect_objects(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Create an image object
    image = types.Image(content=content)

    # Perform object detection on the image
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    # Extract object descriptions
    object_descriptions = [obj.name.lower() for obj in objects]

    return object_descriptions

# Example usage:
image_path = 'sample.jpeg'
human_presence, classification, comment, image_description = classify_image(image_path)
print("Human:", human_presence)
print("Real/Fake or Edited/Not Edited:", classification)
print("Comment:", comment)
print("Image Description:", image_description)
objects_detected = detect_objects(image_path)
print("Objects Detected:", objects_detected)
