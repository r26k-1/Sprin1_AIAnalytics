import pandas as pd
from app.models.mean_imputer import MeanImputer
from app.models.mode_imputer import ModeImputer
from app.models.log_transform import LogTransform
import spacy
import fitz  # PyMuPDF for handling PDFs

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Preprocessing for structured data (CSV/Excel)
def preprocess_structured(file_path, file_type):
    try:
        # Load the data
        df = pd.read_csv(file_path) if file_type == "text/csv" else pd.read_excel(file_path)

        # Example preprocessing pipeline
        mean_imputer = MeanImputer(variables=["col1", "col2"])  # Adjust column names as needed
        mode_imputer = ModeImputer(variables=["col3"])  # Adjust column names as needed
        log_transform = LogTransform(variables=["col4"])  # Adjust column names as needed
        
        # Apply the transformations
        df = mean_imputer.fit(df).transform(df)
        df = mode_imputer.fit(df).transform(df)
        df = log_transform.fit(df).transform(df)

        return df.to_json(orient="split")

    except Exception as e:
        raise ValueError(f"Error processing structured data: {str(e)}")

# Preprocessing for unstructured data (PDF/Text)
def preprocess_unstructured(file_path, file_type):
    try:
        if file_type == "application/pdf":
            # PDF processing
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()

            # Clean text (SpaCy for tokenization and lemmatization)
            doc = nlp(text)
            clean_text = " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

            return clean_text
        elif file_type == "text/plain":
            # Text processing
            with open(file_path, 'r') as f:
                text = f.read()

            doc = nlp(text)
            clean_text = " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

            return clean_text
        else:
            raise ValueError("Unsupported file type for unstructured data.")

    except Exception as e:
        raise ValueError(f"Error processing unstructured data: {str(e)}")
