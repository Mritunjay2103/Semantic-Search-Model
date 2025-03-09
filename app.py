import os
import numpy as np
import torch
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to store our model
model = None

def load_model():
    """Load the pretrained model"""
    global model
    logger.info("Loading model...")
    try:
        # Load the Sentence-BERT model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load model at startup
load_model()

def preprocess_text(text):
    """Preprocess the input text"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    return ""

def compute_similarity(text1, text2):
    """Compute semantic similarity between two texts"""
    try:
        # Preprocess texts
        text1 = preprocess_text(text1)
        text2 = preprocess_text(text2)
        
        # If either text is empty, return 0
        if not text1 or not text2:
            return 0.0
            
        # Encode the texts
        embeddings1 = model.encode([text1], convert_to_tensor=True)
        embeddings2 = model.encode([text2], convert_to_tensor=True)
        
        # Compute cosine similarity
        cosine_scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        similarity = cosine_scores.item()
        
        # Ensure the score is between 0 and 1
        return max(0.0, min(similarity, 1.0))
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        return 0.0

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict similarity between two texts"""
    try:
        # Get input data from request
        data = request.get_json(force=True)
        
        # Extract text pairs
        text1 = data.get("text1", "")
        text2 = data.get("text2", "")
        
        logger.info(f"Received request with text1: {text1[:50]}... and text2: {text2[:50]}...")
        
        # Calculate similarity
        similarity_score = compute_similarity(text1, text2)
        
        # Return result in required format
        return jsonify({"similarity_score": similarity_score})
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint - provide basic info about the API"""
    return """
    <h1>Semantic Similarity API</h1>
    <p>This API measures the semantic similarity between two texts.</p>
    <p>To use it, send a POST request to the /predict endpoint with JSON data in this format:</p>
    <pre>
    {
        "text1": "Your first text here",
        "text2": "Your second text here"
    }
    </pre>
    <p>You'll receive a response like this:</p>
    <pre>
    {
        "similarity_score": 0.85
    }
    </pre>
    <p>The similarity score ranges from 0 (completely dissimilar) to 1 (identical meaning).</p>
    """

if __name__ == '__main__':
    # Get port from environment variable (for Heroku deployment)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)