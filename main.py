from fastapi import FastAPI, UploadFile, File
from typing import List
import io
import uvicorn
from functions import read_pdf_text,create_dataframe_from_text,LDA_view

app = FastAPI()

@app.post("/upload/pdf/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    texts = []
    for file in files:
        pdf_content = await file.read()
        pdf_file = io.BytesIO(pdf_content)
        text = read_pdf_text(pdf_file)
        texts.append({"filename": file.filename, "text": text})

    data = create_dataframe_from_text(text)
    LDA_view(data['Texto'])
    return data

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
