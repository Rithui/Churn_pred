import pickle
import re
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import warnings
import numpy as np

# Suppress specific scikit-learn warnings about version mismatch
warnings.filterwarnings("ignore", category=UserWarning, message=".*Trying to unpickle estimator.*")

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Load the model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model files not found. Make sure model.pkl and vectorizer.pkl are in the same directory as app.py")

# Enhanced spam detection features
def extract_spam_features(text):
    """Extract additional features that might indicate spam"""
    features = {}
    
    # Check for common spam phrases
    spam_phrases = [
        'congratulations', 'winner', 'won', 'prize', 'claim', 
        'free', 'offer', 'limited time', 'click here', 'hurry',
        'opportunity', 'exclusive', 'selected', 'lucky', 'iphone',
        'urgent', 'suspended', 'verify', 'account', 'security',
        'bank', 'credit card', 'password', 'login', 'suspicious'
    ]
    
    text_lower = text.lower()
    
    # Count spam phrases
    features['spam_phrase_count'] = sum(1 for phrase in spam_phrases if phrase in text_lower)
    
    # Check for excessive punctuation
    features['exclamation_count'] = text.count('!')
    
    # Check for suspicious TLDs in URLs
    suspicious_tlds = ['.xyz', '.info', '.top', '.win', '.loan', '.stream']
    urls = re.findall(r'https?://\S+|www\.\S+', text_lower)
    features['suspicious_url_count'] = sum(1 for url in urls if any(tld in url for tld in suspicious_tlds))
    
    # Check for urgency indicators
    urgency_phrases = ['limited time', 'act now', 'hurry', 'expires', 'today only', '24 hours', 'don\'t miss']
    features['urgency_count'] = sum(1 for phrase in urgency_phrases if phrase in text_lower)
    
    return features

# Text preprocessing function
ps = PorterStemmer()
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs but count them first (for feature extraction)
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    url_count = len(urls)
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and apply stemming
    tokens = [ps.stem(word) for word in tokens if word not in stopwords.words('english')]
    
    # Join tokens back into text
    return " ".join(tokens), url_count

def predict_with_enhanced_features(email_text):
    """Make prediction with enhanced spam features"""
    # Transform the text
    transformed_text, url_count = transform_text(email_text)
    
    # Use the vectorizer to transform the text
    vector_input = vectorizer.transform([transformed_text])
    
    # Convert to array
    vector_array = vector_input.toarray()
    
    # Add the num_characters feature (length of original text)
    num_chars = len(email_text)
    
    # Extract additional spam features
    spam_features = extract_spam_features(email_text)
    
    # Create additional features array
    additional_features = np.array([[num_chars]])
    
    # Combine TF-IDF features with additional features
    vector_array = np.hstack((vector_array, additional_features))
    
    # Make prediction using base model
    base_prediction = model.predict(vector_array)[0]
    
    # Get probability scores
    base_probability = model.predict_proba(vector_array)[0]
    
    # Enhanced rule-based detection
    is_likely_spam = (
        spam_features['spam_phrase_count'] >= 3 or
        spam_features['exclamation_count'] >= 3 or
        spam_features['urgency_count'] >= 2 or
        ('free' in email_text.lower() and any(word in email_text.lower() for word in ['iphone', 'prize', 'won', 'winner'])) or
        ('urgent' in email_text.lower() and 'account' in email_text.lower()) or
        ('congratulations' in email_text.lower() and 'winner' in email_text.lower())
    )
    
    # Final prediction (combine model prediction with rule-based detection)
    final_prediction = 1 if (base_prediction == 1 or is_likely_spam) else 0
    
    # Adjust confidence based on rule-based detection
    confidence = base_probability[1] if final_prediction == 1 else base_probability[0]
    if is_likely_spam and final_prediction == 1:
        confidence = max(confidence, 0.85)  # Increase confidence for rule-based spam detection
    
    return final_prediction, confidence, is_likely_spam, spam_features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get email content from form
        email_text = request.form.get('email_text')
        
        # Make prediction with enhanced features
        prediction, confidence, is_rule_based, spam_features = predict_with_enhanced_features(email_text)
        
        # Determine result
        result = "Ham (Not Spam)" if prediction == 0 else "Spam"
        confidence = round(confidence * 100, 2)
        
        # Add explanation for rule-based detection
        explanation = ""
        if is_rule_based and prediction == 1:
            explanation = "This email was detected as spam based on suspicious patterns commonly found in phishing emails."
            
            # Add more detailed explanation
            details = []
            if spam_features['spam_phrase_count'] >= 3:
                details.append(f"Contains {spam_features['spam_phrase_count']} suspicious phrases")
            if spam_features['exclamation_count'] >= 3:
                details.append(f"Contains {spam_features['exclamation_count']} exclamation marks")
            if spam_features['urgency_count'] >= 2:
                details.append("Creates a false sense of urgency")
            if 'free' in email_text.lower() and any(word in email_text.lower() for word in ['iphone', 'prize', 'won', 'winner']):
                details.append("Offers suspicious free items or prizes")
                
            if details:
                explanation += " Suspicious elements: " + ", ".join(details) + "."
        
        return render_template('result.html', 
                              email_text=email_text,
                              result=result,
                              confidence=confidence,
                              explanation=explanation)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for email classification (used by automation script)"""
    data = request.get_json()
    email_text = data.get('email_text', '')
    
    # Make prediction with enhanced features
    prediction, confidence, is_rule_based, spam_features = predict_with_enhanced_features(email_text)
    
    # Determine result
    result = "Ham" if prediction == 0 else "Spam"
    
    return jsonify({
        'result': result,
        'confidence': round(confidence * 100, 2),
        'prediction': int(prediction),
        'rule_based_detection': is_rule_based,
        'spam_features': {
            'spam_phrase_count': spam_features['spam_phrase_count'],
            'exclamation_count': spam_features['exclamation_count'],
            'urgency_count': spam_features['urgency_count']
        }
    })

if __name__ == '__main__':
    app.run(debug=True)