from flask import Flask, request, jsonify
import torch
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

with open("bi_lstm_model_1.pkl", "rb") as f:
    model = pickle.load(f)
model.eval()  

def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet_tokens = tweet.split()  
    return tweet_tokens

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        tweet = data['tweet']
        processed_tweet = preprocess_tweet(tweet)
        tweet_tensor = torch.tensor([processed_tweet])  #
    
        with torch.no_grad():
            outputs = model(tweet_tensor)
            _, predicted_label = torch.max(outputs, 1)  
        
        return jsonify({'predicted_label': int(predicted_label.item())})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
