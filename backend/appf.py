import io
import os
import subprocess
from collections import Counter
import cv2
import imageio
import newspaper
import numpy as np
import tensorflow as tf
from classifiers import Meso4
from finalaudiocode.audiofinal import main
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_file, send_from_directory, session, url_for)
from google.cloud import vision
from google.cloud.vision_v1 import types
from googleapiclient.discovery import build
from keras.preprocessing import image
from pymongo import MongoClient
from scipy.fft import dct
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.inception_v3 import (InceptionV3,
                                                        preprocess_input)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from transformers import (BertForSequenceClassification, BertTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer, pipeline)
from werkzeug.utils import secure_filename
from youtube import process_videos
import requests
from youtube_dl import YoutubeDL
import urllib.request
from urllib.error import HTTPError
from tensorflow.keras.utils import get_file
from io import BytesIO
import tempfile

app = Flask(__name__)

app.static_folder = 'templates'
app.secret_key = 'HKGKJWBEIY%#^@VEHJWV'


# URLs for the models
mymodel_hdf5_url = 'https://myfakeproject.s3.ap-south-1.amazonaws.com/mymodel.hdf5'
res_caffemodel_url = 'https://myfakeproject.s3.ap-south-1.amazonaws.com/res10_300x300_ssd_iter_140000.caffemodel'
deploy_prototxt_url = 'https://myfakeproject.s3.ap-south-1.amazonaws.com/deploy.prototxt'


# Configure MongoDB connection
mongo_uri = "mongodb+srv://webuild:zDEvvvSPzT7ZcUNE@contruction.y5uhaai.mongodb.net/"
client = MongoClient(mongo_uri)
db = client["dface"]
collection = db['fakeimage']


# Route for the home page
@app.route('/')
def home():
    return render_template('front.html')


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Retrieve the user from the database
        users = db.users
        user = users.find_one({'email': email, 'password': password})

        if user:
            session['email'] = user['email']
            session['username'] = user['username']
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            return "Login failed"
    return render_template('login.html')


@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        # Get user data from the signup form
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # Insert the user into the database
        users = db.users
        users.insert_one({'username': username, 'password': password, 'email': email})

        return render_template('login.html')
    
    return render_template('signup.html')
    

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    user_name = session.get('username')
    if 'logged_in' in session and session['logged_in']:
        return render_template('dashboard.html', user_name=user_name)
    else:
        return "Unauthorized. Please log in first."

@app.context_processor
def inject_username():
    # Define a function to make the username available to all templates
    username = session.get('username', None)
    return dict(user_name=username)


@app.route('/news', methods=['GET','POST'])
def detect_fake_news():
    user_input = ""
    
    if request.method == 'POST':
        
        # Load the pre-trained BERT model and tokenizer for fake news detection
        bert_model_name = "bert-base-uncased"
        bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Load a pre-trained T5 model and tokenizer for generating explanations
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

        # Define a text classification pipeline
        classifier = pipeline('sentiment-analysis', model=bert_model, tokenizer=bert_tokenizer)

        # Function to analyze fake news
        def analyze_fake_news(news_text):
            try:
                # Classify the news text
                result = classifier(news_text)
                return result
            except Exception as e:
                return str(e)

        # Function to scrape a news article from a URL
        def scrape_article(url):
            article = newspaper.Article(url)
            article.download()
            article.parse()
            return article.text

        # Function to generate explanations using T5
        def generate_explanation(news_text, classification_result):
            input_text = f"Explain why the news article is classified as {classification_result}: {news_text}"

            # Tokenize and generate an explanation
            input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=200, truncation=True)
            generated_ids = t5_model.generate(input_ids, max_length=200, num_return_sequences=1, early_stopping=False)

            explanation = t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            return explanation

        user_input = request.form['article_url']
        analysis_result = analyze_fake_news(user_input)

        if isinstance(analysis_result, str):
            return render_template('news.html', response=f"Error analyzing fake news: {analysis_result}")
        else:
            label = analysis_result[0]['label']
            confidence = analysis_result[0]['score'] * 100
            confidence_threshold = 70  # Set your desired threshold here
            is_fake = confidence >= confidence_threshold

            labels = ['Real', 'Fake']

            predicted_label = 1 if is_fake else 0

            # Scraping the news article from a URL (you can replace the URL)
            article_url = user_input
            news_text = scrape_article(article_url)

            # Generating an explanation for the classification result
            explanation = generate_explanation(news_text, label)

            return render_template('news.html', news_text=user_input, label=label, confidence=confidence,predicted_label=predicted_label, is_fake=is_fake, explanation=explanation, labels=labels)

    return render_template('news.html')

@app.route('/uploadfakenews', methods=['POST'])
def uploadnews():
        # Get data from the form
        news_text = request.form.get('news_text')
        label = request.form.get('label')
        confidence = request.form.get('confidence')
        predicted_label = request.form.get('predicted_label')
        is_fake = request.form.get('is_fake')
        explanation = request.form.get('explanation')
    

        document = {
            'news_text': news_text,
            'label': label,
            'confidence': confidence,
            'predicted_label': predicted_label,
            'is_fake': is_fake,
            'explanation': explanation
        }

        # Insert the document into the MongoDB colleccction
        collection.insert_one(document)

        return render_template('news.html')


@app.route('/audio', methods=['GET', 'POST'])
def audio():
    if request.method == 'POST':
        # Handle the audio file upload (use the 'request.files' object)
        audio_file = request.files['audio_file']

        # Save the uploaded audio file to a temporary location
        temp_path = 'audio.wav'
        audio_file.save(temp_path)

        # Call your audio classification function
        explanation, transcription = main(temp_path)

        # Render the 'audio.html' template and pass the results
        return render_template('audio.html', classification_result=explanation, transcribed_text=transcription)

    return render_template('audio.html', classification_result=None, transcribed_text=None)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'fakeimage/vision-414807-5c7c9e0314a6.json'


# Route to handle image submission
@app.route('/submit-image', methods=['POST'])
def submit_image():
    
    # S3 bucket URLs for the h5 models
    fake_image_classifier_url = 'https://myfakeproject.s3.ap-south-1.amazonaws.com/fake_image_classifier_model.h5'
    human_classifier_url = 'https://myfakeproject.s3.ap-south-1.amazonaws.com/human_classifier_model_with_vgg16.h5'

    try:
        # Load the fake image classifier model
        with urllib.request.urlopen(fake_image_classifier_url) as f_fake:
            fake_image_classifier_bytes = f_fake.read()
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_fake_model_file:
            temp_fake_model_file.write(fake_image_classifier_bytes)
            temp_fake_model_file.seek(0)
            model_fake_detector = load_model(temp_fake_model_file.name)

        # Load the human classifier model
        with urllib.request.urlopen(human_classifier_url) as f_human:
            human_classifier_bytes = f_human.read()
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_human_model_file:
            temp_human_model_file.write(human_classifier_bytes)
            temp_human_model_file.seek(0)
            model_human_classifier = load_model(temp_human_model_file.name)
        
    except Exception as e:
        print(f"Error loading models from S3: {e}")

    # Initialize the Google Cloud Vision client
    client = vision.ImageAnnotatorClient()


    # Function to preprocess the image for TensorFlow model
    def preprocess_image_tf(img_path):
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        img_array /= 255.0  # Rescale pixel values
        return img_array


    # Function to preprocess the image for Vision API
    def preprocess_image_vision(image_path):
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        # Create an image object
        image = types.Image(content=content)
        return image


    # Function to classify image using TensorFlow model
    def classify_image_tf(img_array):
        prediction = model_human_classifier.predict(img_array)
        return "Human" if prediction >= 0 else "Non-Human"


    # Function to classify image using Vision API
    def classify_image_vision(image):
        # Perform label detection on the image
        response = client.label_detection(image=image)
        labels = response.label_annotations

        # Extract label descriptions
        label_descriptions = [label.description.lower() for label in labels]

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
    # Get the selected option from the request
    selected_option = request.form['selectedOption']

    # Get the uploaded image file
    uploaded_image = request.files['image']

    # Save the uploaded image to a temporary file
    temp_image_path = 'temp_image.jpg'
    uploaded_image.save(temp_image_path)

    # Process the image based on the selected option
    if selected_option == 'vision-api':
        image_data = preprocess_image_vision(temp_image_path)
        result = classify_image_vision(image_data)
    elif selected_option == 'normal':
        img_array = preprocess_image_tf(temp_image_path)
        result = classify_image_tf(img_array)
    else:
        result = "Invalid option selected."

    # Delete the temporary image file
    os.remove(temp_image_path)

    # Return the result
    return jsonify(result)


# Route to render the image upload page
@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        # If it's a POST request, submit the image
        return submit_image()
    else:
        # If it's a GET request, render the image upload page
        return render_template('image.html')




@app.route('/streamlit')
def streamlit():
    subprocess.Popen(['python', '-m', 'streamlit', 'run', 'image.py'])
    return render_template('image.html')

@app.route('/videos/<path:filename>')
def download_file(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/videos/<path:filename>')
def download_vid(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/videos/<path:filename>')
def newsvideo(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/videos/<path:filename>')
def audiogiphy(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/videos/<path:filename>')
def ytgiphy(filename):
    return send_from_directory(app.static_folder, filename)


@app.route('/images/image1')
def get_image1():
    image_directory = 'images'
    return send_file(f'{image_directory}/images.jpeg')

@app.route('/images/image2')
def get_image2():
    image_directory = 'images'
    return send_file(f'{image_directory}/imga.jpg')

@app.route('/images/image3')
def get_image3():
    image_directory = 'images'
    return send_file(f'{image_directory}/audio.jpg')

@app.route('/images/image4')
def get_image4():
    image_directory = 'images'
    return send_file(f'{image_directory}/ani4.png')

@app.route('/images/image5')
def get_image5():
    image_directory = 'images'
    return send_file(f'{image_directory}/realcam.jpg')

@app.route('/workflow/image1')
def get_image():
    image_directory = 'workflow'
    return send_file(f'{image_directory}/image.png')

@app.route('/workflow/image2')
def get_audio():
    image_directory = 'workflow'
    return send_file(f'{image_directory}/a.jpg')

@app.route('/workflow/image3')
def get_video():
    image_directory = 'workflow'
    return send_file(f'{image_directory}/v.png')

@app.route('/workflow/image4')
def get_news():
    image_directory = 'workflow'
    return send_file(f'{image_directory}/news.png')

@app.route('/workflow/image5')
def get_realcam():
    image_directory = 'workflow'
    return send_file(f'{image_directory}/realcam.png')

@app.route('/workflow/image6')
def get_yt():
    image_directory = 'workflow'
    return send_file(f'{image_directory}/youtube.png')

@app.route('/report')
def report():
    return render_template('report.html')


# Route for the fake_news_finder page
@app.route('/fake_news_finder', methods=['GET', 'POST'])
def fake_news_finder():
    if request.method == 'POST':
        keyword = request.form['keyword']
        # Fetch other form data if necessary, like api_key and max_results
        api_key = 'AIzaSyCZcZGyR5fB6K-pqCyvvVjVu_W-xM0Ye3E'
        max_results = int(request.form.get('max_results', 3))  # Default to 3 if not provided
        
        # Call the main function from youtube.py with form inputs
        videos, summaries = process_videos(api_key, keyword, max_results)
        
        # Render the template with the fetched data
        return render_template('fake_news_finder.html', videos=videos, summaries=summaries)

    # If GET request or no form submitted, render the page without data
    return render_template('fake_news_finder.html', videos=None, summaries=None)


# Route for the fake_video_detector page
@app.route('/fake_video_detector/', methods=['GET', 'POST'])
def fake_video_detector():
    if request.method == 'POST':
        if 'video-file' in request.files:
            video_file = request.files['video-file']
            temp_file_path = 'temp_video.mp4'
            video_file.save(temp_file_path)

            # Process the video
            duration = calculate_video_duration(temp_file_path)
            predictions = process_video(temp_file_path, model)

            if predictions:
                classification, confidence = classify_video(predictions)
                result = {
                    'classification': classification,
                    'confidence': confidence,
                    'duration': duration
                }
                return jsonify(result)  # Return the result as JSON
            else:
                return jsonify({'error': 'Error occurred during video processing.'})

        else:
            return jsonify({'error': 'No video file uploaded.'}), 400

    return render_template('fake_video_detector.html')


# Route for the working page
@app.route('/working/')
def working():
    return render_template('working.html')

# Route for the privacy policy page
@app.route('/privacy_policy/')
def privacy_policy():
    return render_template('privacy_policy.html')


meso4_model = None
vgg16_model = None

def load_meso4_model():
    global meso4_model
    if meso4_model is None:
        meso4_model = Meso4()
        meso4_model.load('./weights/Meso4_DF.h5')
        
def load_vgg16_model():
    global vgg16_model
    if vgg16_model is None:
        vgg16_model = VGG16(weights='imagenet', include_top=False)

        
def enhance_image(image, target_size=(224, 224)):
    enhanced_image = cv2.resize(image, target_size)
    return enhanced_image

# Function to extract features from an image
def extract_features_from_image(img):
    img = enhance_image(img)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = vgg16_model.predict(img)
    return features

# Function to extract features from an image or video URL
def extract_features_from_url(url):
    try:
        response = requests.get(url)
        data = response.content

        if url.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = cv2.imdecode(np.asarray(bytearray(data)), cv2.IMREAD_COLOR)
            features = extract_features_from_image(image)
            return features
        elif url.lower().endswith(('.mp4', '.avi', '.mov')):
            video = cv2.VideoCapture(BytesIO(data))
            success, frame = video.read()
            if success:
                features = extract_features_from_image(frame)
                return features
            else:
                print("Error reading video frame")
                return None
        else:
            print("Unsupported file format:", url)
            return None
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return None

# Function to find the best match for a query feature among known people features
def find_best_match(query_features, known_people_features):
    if query_features is None:
        return -1

    similarities = []
    query_features = query_features.flatten()

    for known_features in known_people_features:
        if known_features is not None:
            known_features = known_features.flatten()
            similarities.append(cosine_similarity([query_features], [known_features])[0][0])
        else:
            similarities.append(0)

    best_match_idx = np.argmax(similarities)
    return best_match_idx

@app.route('/fake_v/', methods=['GET', 'POST'])
def index():
    cap = None
    if request.method == 'POST':
        video_path = request.form.get('video_input')
        if 'video_input' in request.files:
            file = request.files['video_input']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            if file:
                filename = secure_filename(file.filename)
                file.save(os.path.join('uploads', filename))
                video_path = os.path.join('uploads', filename)
                cap = cv2.VideoCapture(video_path)

        elif video_path.startswith('https'):
            try:
                print("In url")
                yt = YoutubeDL({'format': 'best'})
                info = yt.extract_info(video_path, download=False)
                if info['duration'] > 600:
                    raise ValueError("Video duration exceeds the limit")
                yt.download([video_path])
                cap = cv2.VideoCapture(video_path)
                print(cap)
            except Exception as e:
                print(f"Error downloading video: {e}")
                return jsonify({'error': str(e)}), 400

        else:
            return jsonify({'error': 'Unsupported input video format'}), 400

        load_vgg16_model()

        # Initialize lists to store known people features and names
        known_people_features = []
        known_people_names = []

        # URLs for the images and videos
        image_url = 'https://res.cloudinary.com/dwmz05ivk/image/upload/v1706551333/Truthtracker/Elon/sf8oyuvt7flda0bvch9l.jpg'
        video_url = 'https://res.cloudinary.com/dwmz05ivk/video/upload/v1706551492/Truthtracker/Elon/dwkyyoevpg9fraba2pse.mp4'

        # Extract features from the image and video
        image_features = extract_features_from_url(image_url)
        video_features = extract_features_from_url(video_url)

        # Append the features to the list of known people features
        known_people_features.append(image_features)
        known_people_features.append(video_features)

        # Check if features were extracted successfully
        if None in known_people_features:
            print("Error: Unable to extract features for one or more known people.")
        else:
            print("Features extracted successfully.")

        # Example of finding the best match for a query feature
        query_features = extract_features_from_url('https://res.cloudinary.com/dwmz05ivk/image/upload/v1706551333/Truthtracker/Elon/sf8oyuvt7flda0bvch9l.jpg')
        best_match_idx = find_best_match(query_features, known_people_features)
        if best_match_idx != -1:
            print("Best match found:", known_people_names[best_match_idx])
        else:
            print("No match found.")

        frame_faces = []

        if cap is not None:
            print("Video capture successful")
            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    print("Video capture unsuccessful")
                    break

                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                    minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face = frame[y:y + h, x:x + w]
                    features = extract_features_from_image(face)
                    best_match_idx = find_best_match(features, known_people_features)
                    matched_name = known_people_names[best_match_idx]

                    frame_faces.append(matched_name)

            cap.release()
            cv2.destroyAllWindows()
            print("Video processing completed")

            if frame_faces:
                most_common_face = Counter(frame_faces).most_common(1)[0]
                recognized_person = most_common_face[0]
            else:
                recognized_person = "No faces detected"

            return jsonify({
                'recognized_person': recognized_person,
                'video_input': video_path,
            })

    return render_template('fakev.html')

@app.route('/analyze_deepfake', methods=['POST'])
def analyze_deepfake():
    data = request.get_json()
    response = data.get('response')
    video_path = data.get('video_input')

    print("response:", response)
    print("video_path:", video_path)


    if response == "yes":
        recognized_person = data.get('recognized_person')
        print("recognized_person:", recognized_person)
        recognized_person_video_path = '1027 (1)(2).mp4'

        def predict_deepfake(video_path):
            video_frames = []

            # Capture frames from the video
            video_capture = cv2.VideoCapture(video_path)
            while True:
                success, frame = video_capture.read()
                if not success:
                    break
                video_frames.append(frame)

            # Perform Meso4 predictions
            predictions = []
            for frame in video_frames:
                # Preprocess the frame
                frame = cv2.resize(frame, (256, 256))
                frame = np.expand_dims(frame, axis=0)
                frame = frame / 255.0  # Normalize

                # Make predictions
                prediction = meso4_model.predict(frame)
                predictions.append(prediction)

            # Calculate an average prediction score
            average_score = np.mean(predictions)

            return average_score

        # Analyze if the recognized person's video is real or fake
        deepfake_score = predict_deepfake(recognized_person_video_path)

        if deepfake_score < 0.5:
            result = f'The recognized person video is likely real (Meso4 Score: {deepfake_score})'
        else:
            result = f'The recognized person video is likely fake (Meso4 Score: {deepfake_score})'
    else:
        video_path = request.form.get('video_input')
        print("video-path:",video_path)
        print("In response No")
        # Analyze the entire video

        def predict_deepfake(video_path):
            video_frames = []

            # Capture frames from the video
            video_capture = cv2.VideoCapture(video_path)
            while True:
                success, frame = video_capture.read()
                if not success:
                    break
                video_frames.append(frame)

            # Perform Meso4 predictions
            predictions = []
            for frame in video_frames:
                # Preprocess the frame
                frame = cv2.resize(frame, (256, 256))
                frame = image.img_to_array(frame)
                frame = np.expand_dims(frame, axis=0)
                frame = frame / 255.0  # Normalize

                # Make predictions
                prediction = meso4_model.predict(frame)
                predictions.append(prediction)

            # Calculate an average prediction score
            average_score = np.mean(predictions)

            return average_score

        entire_video_score = predict_deepfake(video_path)

        if entire_video_score < 0.5:
            result = f'The entire video is likely real (Meso4 Score: {entire_video_score})'
        else:
            result = f'The entire video is likely fake (Meso4 Score: {entire_video_score})'

    return jsonify({"result": result})


def calculate_video_duration(video_path):
    try:
        video = imageio.get_reader(video_path, 'ffmpeg')
    except Exception as e:
        print(f"Error: Unable to open the video file. {e}")
        return None

    duration = video.get_meta_data()['duration']
    video.close()

    return duration

def process_video(video_path, model):
    try:
        video = imageio.get_reader(video_path, 'ffmpeg')
    except Exception as e:
        print(f"Error: Unable to open the video file. {e}")
        return []

    frame_count = len(video)
    frame_height, frame_width = video.get_meta_data()['source_size'][:2]

    print(f"Video info: {frame_count} frames, {frame_width}x{frame_height} resolution")

    predictions = []

    try:
        for i, frame in enumerate(video):
            # Resize and preprocess the frame
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype(np.float32) / 255.0
            frame = np.expand_dims(frame, axis=0)
            frame = preprocess_input(frame)

            # Make a prediction
            prediction = model.predict(frame)
            predictions.append(prediction[0][0])  # Append the prediction result (fake score)

    except Exception as e:
        print(f"Error occurred during video processing. {e}")
        return []

    return predictions

def classify_video(predictions, threshold=0.5):
    if not predictions:
        return 'unknown', 0

    fake_count = sum(1 for score in predictions if score > threshold)
    real_count = len(predictions) - fake_count

    fake_score = fake_count / len(predictions)
    real_score = real_count / len(predictions)

    if fake_score > real_score:
        return 'fake', fake_score
    else:
        return 'real', real_score
    
    
# Define the route to handle form submissions
@app.route('/report', methods=['POST'])
def report_user():
    if request.method == 'POST':
        # Get the submitted report user
        report_user = request.form.get('reportUser')

        # Save the report user to a file (you can use any format you prefer)
        with open('report_users.txt', 'a') as file:
            file.write(report_user + '\n')

        return "Report submitted successfully."
    
    
# Function to load the HDF5 model directly from URL
def load_hdf5_model_from_url(url):
    try:
        model_path = tf.keras.utils.get_file("mymodel.hdf5", origin=url)
        model = tf.keras.models.load_model(model_path)
        return model
    except HTTPError as e:
        print(f"Error loading HDF5 model from URL: {e}")
        return None

# Function to load the Caffe model directly from URL
def load_caffe_model_from_url(prototxt_url, caffemodel_url):
    try:
        prototxt = urllib.request.urlopen(prototxt_url).read().decode('utf-8')
        caffemodel = urllib.request.urlopen(caffemodel_url).read()
        prototxt_file = tempfile.NamedTemporaryFile(delete=False)
        caffemodel_file = tempfile.NamedTemporaryFile(delete=False)
        prototxt_file.write(prototxt.encode('utf-8'))
        caffemodel_file.write(caffemodel)
        prototxt_file.close()
        caffemodel_file.close()
        net = cv2.dnn.readNetFromCaffe(prototxt_file.name, caffemodel_file.name)
        return net
    except HTTPError as e:
        print(f"Error loading Caffe model from URL: {e}")
        return None

    
@app.route('/realcam', methods=['GET','POST'])
def real_cam():
    if request.method == 'POST':

        # Load HDF5 model directly from URL
        spoof_detection_model = load_hdf5_model_from_url(mymodel_hdf5_url)

        # Load Caffe model directly from URL
        face_net = load_caffe_model_from_url(deploy_prototxt_url, res_caffemodel_url)

        # Check if models were loaded successfully
        if spoof_detection_model is None or face_net is None:
            print("Error: Unable to load one or more models.")
        else:
            print("Models loaded successfully.")
        
        frame_blob = request.files['frame'].read()
        nparr = np.frombuffer(frame_blob, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        # Set a minimum confidence threshold for face detection
        confidence_threshold = 0.5
        
        results=[]
        
        # Perform face detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 117, 123))
        face_net.setInput(blob)
        
        try:
            detections = face_net.forward()

            # Process each detected face
            for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    # Filter out weak detections
                    if confidence > confidence_threshold:
                        # Extract the face region
                        box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                        (startX, startY, endX, endY) = box.astype(int)
                        face = frame[startY:endY, startX:endX]
                        
                        # Determine label and color based on confidence
                        if confidence > 0.90:
                            label = "Real"
                            color = "green"
                        else:
                            label = "Spoof"
                            color = "red"
                            

            cv2.destroyAllWindows()

            results.append({
                        "startX": int(startX),
                        "startY": int(startY),
                        "endX": int(endX),
                        "endY": int(endY),
                        "label": label,
                        "color": color
                    })

            return jsonify(results=results)
        except Exception as e:
            print(f"An error occurred: {e}")
            return jsonify(error=str(e))

    return render_template('realcam.html')


if __name__ == '__main__':
    # Load the pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add a new output layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    # Create the new model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.001), metrics=['accuracy'])

    app.run(debug=True)
