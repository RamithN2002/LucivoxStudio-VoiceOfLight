from pypdf import PdfReader
from docx import Document 

def load_file(file_path):

    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

        
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text
        return text
    
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as file:
            text = file.read()
            return text
    else:
        raise ValueError("Unsupported file format")

