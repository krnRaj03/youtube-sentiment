import os
import io
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend before pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from googleapiclient.discovery import build  # Add this import for YouTube API

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
print("YouTube API Key loaded:", bool(YOUTUBE_API_KEY))

if not YOUTUBE_API_KEY:
    raise ValueError("❌ YOUTUBE_API_KEY not found in environment. Check your .env file.")

# --------------------------
# Flask app setup
# --------------------------
app = Flask(__name__)
CORS(app)

# --------------------------
# Preprocessing function
# --------------------------
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# --------------------------
# Load MLflow model + vectorizer
# --------------------------
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("http://ec2-35-172-221-238.compute-1.amazonaws.com:5000/")  
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "../tfidf_vectorizer.pkl")

# --------------------------
# Routes
# --------------------------
@app.route('/')
def home():
    return f"Welcome to our Flask API. YT API Key Loaded: {bool(YOUTUBE_API_KEY)}"

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        preprocessed_comments = [preprocess_comment(c) for c in comments]

        transformed = vectorizer.transform(preprocessed_comments).toarray()
        feature_names = vectorizer.get_feature_names_out()
        input_df = pd.DataFrame(transformed, columns=feature_names)

        predictions = model.predict(input_df).tolist()
        predictions = [str(p) for p in predictions]
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    response = [
        {"comment": c, "sentiment": s, "timestamp": t}
        for c, s, t in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(c) for c in comments]
        input_df = pd.DataFrame({'Review': preprocessed_comments})
        predictions = model.predict(input_df).tolist()
        predictions = [str(p) for p in predictions]
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    response = [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
    return jsonify(response)



# New route to fetch comments 
@app.route('/fetch_comments', methods=['GET'])
def fetch_comments():
    video_id = request.args.get('videoId')
    if not video_id:
        return jsonify({"error": "No videoId provided"}), 400

    try:
        # Build YouTube API client with loaded key
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        comments = []
        next_page_token = None
        max_comments = 500  # Limit as per your original code
        
        while len(comments) < max_comments:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            ).execute()
            
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'text': comment['textOriginal'],
                    'timestamp': comment['publishedAt'],
                    'authorId': comment.get('authorChannelId', {}).get('value', 'Unknown')
                })
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        
        return jsonify({"comments": comments})
    except Exception as e:
        app.logger.error(f"Error fetching comments: {str(e)}")
        return jsonify({"error": f"Failed to fetch comments: {str(e)}"}), 500



@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'color': 'w'})
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed = [preprocess_comment(c) for c in comments]
        text = ' '.join(preprocessed)
        wordcloud = WordCloud(
            width=800, height=400, background_color='black',
            colormap='Blues', stopwords=set(stopwords.words('english')), collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for v in [-1, 0, 1]:
            if v not in monthly_percentages.columns:
                monthly_percentages[v] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        for v in [-1, 0, 1]:
            plt.plot(monthly_percentages.index, monthly_percentages[v],
                     marker='o', linestyle='-', label=sentiment_labels[v], color=colors[v])
        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

# --------------------------
# Start Flask
# --------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
