from pydantic import BaseModel
from typing import Optional, List
from PyPDF2 import PdfReader

#======DTOs (Data Transfer Objects) ======
class userPrompts(BaseModel):
    prompt: str = None

class DataUpload(BaseModel):
    fileContent: Optional[str] = None  # Content of the file to upload (the extracted text)
    fileName: Optional[str] = None   # Name of the file to be uploaded.
    prompt: Optional[userPrompts] = None # Nested prompt object

#========================================

#=======PDF Processor=====================
class PdfProcessor:
    def __init__(self):
        pass

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            text=""
            reader=PdfReader(pdf_path)
            num_pages=len(reader.pages)
            for page_num in range(num_pages):
                page=reader.pages[page_num]
                text+=page.extract_text()+"\n\n--- Page Break ---\n\n"
            if not text.strip():
                return "Error: PDF was successfully"
            return text.strip()
        except FileNotFoundError:
            return f"File not found at path: {pdf_path}"
        except Exception as e:
            return f"An error occurred during PDF extraction: {e}"
    
    def process_and_create_upload_dto(self, pdf_path: str, file_name: str, user_prompt: Optional[str] = None) -> DataUpload:
        try:
            extracted_text = self._extract_text_from_pdf(pdf_path)
            prompt_dto = None
            if user_prompt:
                try:
                    prompt_dto = userPrompts(prompt=user_prompt)
                except Exception as e:
                    print(f"Error validating user prompt: {e}")
                    prompt_dto = None 

            data_upload = DataUpload(
                fileContent=extracted_text,
                fileName=file_name,
                prompt=prompt_dto
            )
            return data_upload

        except Exception as e:
            print(f"An error occurred during processing: {e}")
            return DataUpload(fileContent=None, fileName=file_name, prompt=None)
