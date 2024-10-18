import pandas as pd
from app.models.mean_imputer import MeanImputer
from app.models.mode_imputer import ModeImputer
from app.models.log_transform import LogTransform
import spacy
import fitz  # PyMuPDF for handling PDFs
from docx import Document  # For handling .docx files
import os

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text_with_spacy(text):
    """Utility function to clean text using SpaCy."""
    doc = nlp(text)
    clean_text = " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
    return clean_text

def preprocess_csv_data(file_path):
    """Preprocess CSV or Excel file and return preprocessed JSON."""
    try:
        # Automatically infer file type from file extension
        _, file_extension = os.path.splitext(file_path)
        file_type = "text/csv" if file_extension == '.csv' else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # for .xlsx

        # Read CSV or Excel file into DataFrame
        df = pd.read_csv(file_path) if file_type == "text/csv" else pd.read_excel(file_path)

        # Example preprocessing pipeline
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Initialize imputers and transformers if applicable
        if numeric_cols:
            mean_imputer = MeanImputer(variables=numeric_cols)
            log_transform = LogTransform(variables=numeric_cols)
            df = mean_imputer.fit(df).transform(df)
            df = log_transform.fit(df).transform(df)
        
        if categorical_cols:
            mode_imputer = ModeImputer(variables=categorical_cols)
            df = mode_imputer.fit(df).transform(df)

        # Return preprocessed data as JSON
        return df.to_json(orient="split")

    except FileNotFoundError as fnf_error:
        raise ValueError(f"File not found: {fnf_error}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {file_path}")
    except Exception as e:
        raise ValueError(f"Error processing CSV/Excel file: {str(e)}")

def preprocess_pdf_data(file_path):
    """Extract text from a PDF file and preprocess it using SpaCy."""
    try:
        text = ""
        # Open the PDF and extract text page by page
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()

        if not text.strip():
            raise ValueError("PDF file contains no readable text.")

        clean_text = clean_text_with_spacy(text)
        return clean_text

    except FileNotFoundError as fnf_error:
        raise ValueError(f"File not found: {fnf_error}")
    except Exception as e:
        raise ValueError(f"Error processing PDF file: {str(e)}")

def preprocess_docx_data(file_path):
    """Preprocess .docx file and return its content as cleaned text."""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])

        if not text.strip():
            raise ValueError("DOCX file contains no readable text.")

        clean_text = clean_text_with_spacy(text)
        return clean_text

    except FileNotFoundError as fnf_error:
        raise ValueError(f"File not found: {fnf_error}")
    except Exception as e:
        raise ValueError(f"Error processing DOCX file: {str(e)}")

def preprocess_text(file_path):
    """Preprocess plain text files and return cleaned text."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if not text.strip():
            raise ValueError("Text file is empty.")

        clean_text = clean_text_with_spacy(text)
        return clean_text

    except FileNotFoundError as fnf_error:
        raise ValueError(f"File not found: {fnf_error}")
    except Exception as e:
        raise ValueError(f"Error processing text file: {str(e)}")
