import os
import pandas as pd
import fitz  # PyMuPDF for handling PDF files
from docx import Document  # For handling .docx files
from app.utils.preprocess import preprocess_csv_data, preprocess_pdf_data, preprocess_text, preprocess_docx_data

def load_docx_file(file_path):
    """Load .docx file and return its content as plain text."""
    try:
        return preprocess_docx_data(file_path)  # Directly use the preprocessing function
    except Exception as e:
        raise ValueError(f"Error loading .docx file: {str(e)}")

def load_file(file_type, file_path):
    """Load a file based on its type and return its processed content."""
    try:
        if file_type == 'text/csv':
            # Process CSV file
            return preprocess_csv_data(file_path)
        elif file_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                            'application/vnd.ms-excel']:
            # Process Excel file
            try:
                df = pd.read_excel(file_path, engine='openpyxl')  # Use openpyxl for better compatibility
                return df.to_dict(orient='records')  # Return as a list of dictionaries
            except Exception as e:
                raise ValueError(f"Error processing Excel file: {str(e)}")
        elif file_type == 'application/pdf':
            # Process PDF file
            return preprocess_pdf_data(file_path)
        elif file_type == 'text/plain':
            # Process text file
            return preprocess_text(file_path)  # Pass file path directly
        elif file_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            # Process .doc or .docx file
            if file_type == 'application/msword':  # .doc files
                raise ValueError("Processing of .doc files is not implemented in this example.")
            else:  # .docx files
                return load_docx_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")
