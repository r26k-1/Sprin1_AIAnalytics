# FastAPI Text and Structured Data Preprocessing Application

This project provides a FastAPI-based application for handling both structured data (CSV, Excel) and unstructured data (PDF, Text) uploads. The app performs various preprocessing tasks such as missing data imputation, log transformations, and text analysis using NLP techniques.

## Features

- File uploads for structured data (CSV, Excel) and unstructured data (PDF, Text)
- Structured data preprocessing: mean imputation, mode imputation, log transformation
- Unstructured data processing: PDF and text handling with SpaCy for tokenization and lemmatization
- Integration with MongoDB for storing preprocessed data

## Project Structure

```bash
fastapi_app/
│
├── app/
│   ├── __init__.py
│   ├── main.py               # Main FastAPI app code
│   ├── utils.py              # Utilities for file handling and preprocessing
│   ├── preprocessing.py      # Preprocessing logic (structured and unstructured)
│   ├── models/               # Model classes (MeanImputer, ModeImputer, LogTransform)
│   ├── static/
│   │   └── styles.css        # CSS for frontend
│   └── templates/            # HTML templates for file upload and results
├── uploads/                  # Directory for uploaded files
├── .env                      # Environment variables (MongoDB URI)
├── requirements.txt          # Dependencies
└── README.md                 # Project description and setup instructions
