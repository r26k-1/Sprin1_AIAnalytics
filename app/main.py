from fastapi import FastAPI, UploadFile, File, HTTPException, Request 
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from app.utils.file_loader import load_file
from app.models.sentiment_analyzer import SentimentAnalyzer
from app.models.summarizer import Summarizer
import os
import logging
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")  # Ensure this path is correct

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB client setup
client = MongoClient("mongodb://localhost:27017/")
db = client['file_data']
collection = db['documents']

# Ensure the uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize SentimentAnalyzer and Summarizer
sentiment_analyzer = SentimentAnalyzer()
summarizer = Summarizer()

# Load tokenizer for transformer-based models
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
MAX_TOKENS = 512  # Adjust based on the model's token limit

def chunk_text(text, max_tokens=512):
    """Utility to chunk large text into smaller token chunks."""
    tokens = tokenizer.encode(text, truncation=True)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks

def process_large_text(text):
    """Process large text by chunking, then summarize and analyze sentiment."""
    chunks = chunk_text(text)
    summaries = []
    sentiments = []
    for chunk in chunks:
        summary = summarizer.summarize(chunk)  # Process each chunk
        sentiment = sentiment_analyzer.analyze_with_transformers(chunk)
        summaries.append(summary)
        sentiments.append(sentiment)
    return summaries, sentiments

# HTML Page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logging.info("Rendering the home page.")
    return templates.TemplateResponse("upload.html", {"request": request, "summary": "", "sentiment": ""})

# API to upload and preprocess files
@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    logging.info(f"Received file: {file.filename} of type: {file.content_type}")

    # Validate file type
    if file.content_type not in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/csv",
        "text/plain",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "application/msword"  # .doc
    ]:
        logging.warning(f"Unsupported file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logging.info(f"File saved to {file_path}")

        # Use utility function to load and preprocess the file
        preprocessed_data = load_file(file.content_type, file_path)

        # Extract the cleaned text
        clean_text = preprocessed_data['clean_text'] if isinstance(preprocessed_data, dict) else preprocessed_data

        # Store cleaned content in MongoDB
        document = {
            "filename": file.filename,
            "content": clean_text
        }
        result = collection.insert_one(document)
        logging.info(f"Document inserted into MongoDB with ID: {result.inserted_id}")

        # Check if the text exceeds max tokens
        tokens = tokenizer.encode(clean_text)
        if len(tokens) > MAX_TOKENS:
            logging.warning(f"Text exceeds max token limit of {MAX_TOKENS}. Chunking the input text.")
            summaries, sentiments = process_large_text(clean_text)
            summary = " ".join(summaries)
            transformers_result = {"chunked_analysis": sentiments}
        else:
            summary = summarizer.summarize(clean_text)
            transformers_result = sentiment_analyzer.analyze_with_transformers(clean_text)

        # Perform sentiment analysis using both transformers and TextBlob
        textblob_result = sentiment_analyzer.analyze_with_textblob(clean_text)

        sentiment = {
            "transformers": transformers_result,
            "textblob": textblob_result
        }

        logging.info(f"Sentiment analysis and summarization completed for file: {file.filename}")

        return templates.TemplateResponse("upload.html", {
            "request": request,
            "summary": summary,
            "sentiment": sentiment
        })

    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
