RAG_ML_ASSISTANT
A Streamlit-based document analysis application that combines Retrieval-Augmented Generation (RAG) with Machine Learning for intelligent document querying.
Features

📄 PDF Document Upload: Upload multiple PDF files
🔍 RAG-based Q&A: Ask questions about your documents using advanced retrieval
🤖 ML Analysis: Optional sentiment/tone classification
🚀 Easy to Use: Simple web interface powered by Streamlit

Installation
1. Clone or Download the Project
bashgit clone <your-repo-url>
cd rag-ml-assistant
2. Create Virtual Environment (Recommended)
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bashpip install -r requirements.txt
4. Set Up HuggingFace API Token
You need a HuggingFace API token to use the LLM:

Go to https://huggingface.co/settings/tokens
Create a new token (read access is sufficient)
Set it as an environment variable:

Linux/Mac:
bashexport HUGGINGFACEHUB_API_TOKEN="your_token_here"
Windows (Command Prompt):
cmdset HUGGINGFACEHUB_API_TOKEN=your_token_here
Windows (PowerShell):
powershell$env:HUGGINGFACEHUB_API_TOKEN="your_token_here"
Usage
Running the Application
bashstreamlit run app.py
The application will open in your default web browser at http://localhost:8501
Using the App

Upload PDFs: Click the file uploader and select one or more PDF documents
Wait for Processing: The app will extract text and create the RAG pipeline
Ask Questions: Type your question in the text input box
View Results: See the AI-generated answer and optional ML analysis

Training the ML Model (Optional)
The ML analyzer requires training before it can predict sentiment. Here's how to train it:
pythonfrom ml_analyzer import DocumentAnalyzer

# Sample training data
texts = [
    "This is excellent work!",
    "This is terrible and disappointing.",
    "Great job on this project.",
    "Poor quality and bad service."
]
labels = ["positive", "negative", "positive", "negative"]

# Train the model
analyzer = DocumentAnalyzer()
analyzer.train(texts, labels)
Project Structure
rag-ml-assistant/
├── app.py              # Main Streamlit application
├── rag_pipeline.py     # RAG pipeline using LangChain
├── ml_analyzer.py      # ML-based document analyzer
├── requirements.txt    # Python dependencies
└── README.md          # This file
Key Improvements in This Version

✅ Fixed emoji encoding issues
✅ Added proper error handling
✅ Improved import statements for latest LangChain
✅ Added comprehensive docstrings
✅ Enhanced ML model with predict_proba method
✅ Better file existence checks
✅ Added environment variable validation
✅ Cleaner code formatting

Troubleshooting
Issue: "HUGGINGFACEHUB_API_TOKEN not set"
Solution: Make sure you've set the HuggingFace API token as an environment variable
Issue: "Model file not found"
Solution: The ML model needs to be trained first. See the "Training the ML Model" section
Issue: PDFs not extracting text
Solution: Ensure PDFs contain actual text (not just scanned images)
Issue: Slow response times
Solution: First-time model loading can be slow. Subsequent queries will be faster
Dependencies

langchain: Framework for building LLM applications
faiss-cpu: Vector similarity search
sentence-transformers: Text embeddings
transformers: HuggingFace model access
scikit-learn: ML classification
streamlit: Web interface
pdfplumber: PDF text extraction

License
MIT License - feel free to use and modify
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.