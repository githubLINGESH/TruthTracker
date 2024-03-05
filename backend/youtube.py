from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from collections import Counter
from datetime import datetime
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel


if "en_core_web_sm" not in spacy.util.get_installed_models():
    os.system("python -m spacy download en_core_web_sm")

# Load the 'en_core_web_sm' model
nlp = spacy.load("en_core_web_sm")

# Initialize YouTube API with API key
def initialize_youtube(api_key):
    if not api_key:
        raise ValueError("YouTube API key not found.")
    return build('youtube', 'v3', developerKey=api_key)


# Fetch comments for a specific video
def fetch_video_comments(youtube, video_id):
    try:
        comments = []
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        )
        while request:
            response = request.execute()
            for comment in response["items"]:
                text = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(text)
            request = youtube.commentThreads().list_next(request, response)
        return comments
    except Exception as e:
        print("Error fetching video comments:", str(e))
        return []


# Analyze sentiment of comments
def analyze_sentiment(comments):
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for comment in comments:
            sentiment = analyzer.polarity_scores(comment)
            if sentiment['compound'] >= 0.05:
                sentiment_scores['positive'] += 1
            elif sentiment['compound'] <= -0.05:
                sentiment_scores['negative'] += 1
            else:
                sentiment_scores['neutral'] += 1
        
        return sentiment_scores
    except Exception as e:
        print("Error analyzing sentiment:", str(e))
        return {'positive': 0, 'neutral': 0, 'negative': 0}

# Analyze common themes in comments
def analyze_common_themes(comments):
    try:
        # Process each comment and extract nouns and noun phrases
        themes = []
        for comment in comments:
            doc = nlp(comment)
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"]:
                    themes.append(token.text)
                elif token.pos_ == "NOUN_CHUNK":
                    themes.append(token.text)
        
        # Count occurrences of themes
        theme_counter = Counter(themes)
        
        # Get the most common themes
        common_themes = theme_counter.most_common(3)  # Get top 3 most common themes
        return [theme[0] for theme in common_themes]
    except Exception as e:
        print("Error analyzing common themes:", str(e))
        return []

# Main function to execute the workflow
def execute_tool(api_key, keyword, max_results):
    try:
        # Initialize YouTube API
        youtube = initialize_youtube(api_key)
        
        # Fetch YouTube data based on user input
        videos = fetch_youtube_data(youtube, keyword, max_results)
        
        # Analyze sentiment and common themes for each video
        for video in videos:
            # Fetch comments for the video
            comments = fetch_video_comments(youtube, video['video_id'])
            
            # Perform sentiment analysis
            sentiment_scores = analyze_sentiment(comments)
            video['sentiment_scores'] = sentiment_scores  # Add sentiment scores to video data
            
            # Analyze common themes
            common_themes = analyze_common_themes(comments)
            video['common_themes'] = common_themes

            # Add timestamp
            video['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Add sentiment summary
            total_comments = sum(sentiment_scores.values())
            if total_comments > 0:
                sentiment_summary = {
                    'positive_percentage': sentiment_scores['positive'] / total_comments * 100,
                    'neutral_percentage': sentiment_scores['neutral'] / total_comments * 100,
                    'negative_percentage': sentiment_scores['negative'] / total_comments * 100
                }
            else:
                sentiment_summary = {
                    'positive_percentage': 0,
                    'neutral_percentage': 0,
                    'negative_percentage': 0
                }
            video['sentiment_summary'] = sentiment_summary
        
        return videos
    except Exception as e:
        print("Error executing tool:", str(e))
        return []
    

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate summary
def generate_summary(video_data):
    summaries = []
    max_input_length = 1000  # Define the maximum length for input text
    
    for video in video_data:
        # Combine title, description, and comments into one input text
        input_text = f"Title: {video['video_title']}\nDescription: {video['description']}\nComments:\n" + "\n".join(video['comments'])
        
        # Split input text into chunks of maximum length
        input_chunks = [input_text[i:i + max_input_length] for i in range(0, len(input_text), max_input_length)]
        
        # Generate summary text for each chunk
        for chunk in input_chunks:
            input_ids = tokenizer.encode(chunk, return_tensors="pt")
            
            # Generate summary text
            summary_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, early_stopping=True)
            
            # Decode summary text
            summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Append to summaries list
            summaries.append(summary_text)
    
    return summaries

    
    
def fetch_youtube_data(youtube, keyword, max_results):
    try:
        search_response = youtube.search().list(
            part="snippet",
            q=keyword,
            type="video",
            maxResults=max_results,
        ).execute()

        videos = []
        for item in search_response.get("items", []):
            video_title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            channel_title = item["snippet"]["channelTitle"]
            published_at = item["snippet"]["publishedAt"]
            description = item["snippet"]["description"]  # Added description

            video_response = youtube.videos().list(
                part="statistics",
                id=video_id
            ).execute()

            view_count = int(video_response["items"][0]["statistics"].get("viewCount", 0))
            like_count = int(video_response["items"][0]["statistics"].get("likeCount", 0))
            comment_count = int(video_response["items"][0]["statistics"].get("commentCount", 0))

            # Fetch comments for the video
            comments = fetch_video_comments(youtube, video_id)

            videos.append({
                'video_title': video_title,
                'video_id': video_id,
                'channel_title': channel_title,
                'published_at': published_at,
                'description': description,  # Added description
                'view_count': view_count,
                'like_count': like_count,
                'comment_count': comment_count,
                'comments': comments  # Added comments
            })

        return videos
    except Exception as e:
        print("Error fetching YouTube data:", str(e))
        return []


# At the end of youtube.py, define the main function
def process_videos(api_key, keyword, max_results=3):
    print("API Key",api_key)
    youtube = initialize_youtube(api_key)
    videos = fetch_youtube_data(youtube, keyword, max_results)
    videos = execute_tool(api_key, keyword, max_results)
    summaries = generate_summary(videos)
    return videos, summaries
