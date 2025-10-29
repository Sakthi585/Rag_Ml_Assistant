# ml_analyzer.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os


class DocumentAnalyzer:
    """
    A simple ML-based document analyzer for sentiment/tone classification.
    """
    
    def __init__(self, model_path="ml_model.pkl"):
        self.model_path = model_path
        self.vectorizer = None
        self.model = None
        self._load_model()  # Try to load the model on initialization

    def _load_model(self):
        """Loads the model and vectorizer from disk if they exist."""
        if os.path.exists(self.model_path):
            try:
                # Load pre-trained model and vectorizer
                self.vectorizer, self.model = joblib.load(self.model_path)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Warning: Could not load model. {e}")
                self._initialize_new_model()
        else:
            # If no model file exists, initialize new ones
            self._initialize_new_model()
            
    def _initialize_new_model(self):
        """Initializes new, untrained model objects."""
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression(max_iter=1000)
    
    def train(self, texts, labels):
        """
        Train the classifier on provided texts and labels.
        
        Args:
            texts (list): List of text samples
            labels (list): Corresponding labels for classification
        """
        # Ensure model objects are initialized
        if self.vectorizer is None or self.model is None:
             self._initialize_new_model()
             
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        joblib.dump((self.vectorizer, self.model), self.model_path)
        print(f"Model trained and saved to {self.model_path}")
    
    def predict(self, text):
        """
        Predict the class/sentiment of input text.
        """
        # Check if the model is trained and loaded
        if self.model is None or not hasattr(self.model, 'classes_'):
            raise FileNotFoundError(
                f"Model file '{self.model_path}' not found or model is not trained. "
                "Please train the model first."
            )
        
        # Use the model loaded in memory
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]
    
    def predict_proba(self, text):
        """
        Get prediction probabilities for input text.
        """
        if self.model is None or not hasattr(self.model, 'classes_'):
            raise FileNotFoundError(
                f"Model file '{self.model_path}' not found or model is not trained. "
                "Please train the model first."
            )
        
        # Use the model loaded in memory
        X = self.vectorizer.transform([text])
        return self.model.predict_proba(X)[0]