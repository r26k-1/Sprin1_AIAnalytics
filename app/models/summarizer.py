from transformers import pipeline

class Summarizer:
    def __init__(self):
        # Load the summarization pipeline
        self.model = pipeline("summarization", model="facebook/bart-large-cnn")


    def summarize(self, text):
    # Ensure text is a string and check its length
     if not isinstance(text, str):
        raise ValueError("Input must be a string.")

     if len(text) < 1:  # Adjust the threshold as needed
        return "Input text is too short to summarize."

    # Perform summarization on the provided text
     results = self.model(text)
     return results[0]['summary_text']

        

    
