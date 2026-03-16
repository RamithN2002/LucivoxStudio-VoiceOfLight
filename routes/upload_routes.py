from fastapi import APIRouter, UploadFile, File, Depends
import os
from utils.file_loader import load_file
from rag.parent_document_ingestion import ingest_document
from utils.memory import clear_memory
from auth.auth_deps import get_current_user

router = APIRouter()

UPLOAD_DIR = "Documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user) 
):
    username = current_user["username"]
    scoped_filename = f"{username}__{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, scoped_filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = load_file(file_path)

    clear_memory()
    ingest_document(text, scoped_filename)

    return {
        "status": f"{file.filename} indexed successfully",
        "scoped_filename": scoped_filename,
        "original_filename": file.filename,
    }


@router.get("/my-documents")
async def get_my_documents(current_user: dict = Depends(get_current_user)):
    """Returns list of documents uploaded by the current user."""
    username = current_user["username"]
    prefix = f"{username}__"

    docs = []
    for fname in os.listdir(UPLOAD_DIR):
        if fname.startswith(prefix):
            docs.append({
                "scoped_filename": fname,
                "display_name": fname[len(prefix):]  
            })

    return {"documents": docs}