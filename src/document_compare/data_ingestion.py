import sys
import uuid
from pathlib import Path
import fitz
from datetime import datetime,timezone
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException


class DocumentIngestion:
    """
    Handles saving, reading, and combining of PDFs for comparison with session-based versioning.
    """

    def __init__(self, base_dir: str = "data\\document_compare", session_id=None):
        self.log = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)
        self.session_id = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)

        self.log.info("DocumentComparator initialized", session_path=str(self.session_path))

        pass

    def save_uploaded_files(self,reference_file,actual_file):

        try:
            self.delete_existing_files()
            self.log.info("Existing file deleted successsfully.")

            ref_path = self.base_dir/reference_file.name
            act_path = self.base_dir/actual_file.name

            if not reference_file.name.endswith(".pdf") or not actual_file.name.endswith(".pdf"):
                raise ValueError("Only PDF files are allowed.")
            
            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())

            with open(act_path, "wb") as f:
                f.write(actual_file.getbuffer())

            self.log.info("Files Saved", reference=str(ref_path), actual = str(act_path))
            return ref_path, act_path

        except Exception as e:
            self.log.error("Error reading PDF", error=str(e))
            raise DocumentPortalException("Error reading PDF", sys)


    def read_pdf(self,pdf_path: Path)->str:

        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encryptted : {pdf_path.name}")
                all_text = []
                for page_num in range(doc.page_count):
                    page=doc.load_page(page_num)
                    text = page.get_text()

                    if text.strip():
                        all_text.append(f"\n --- Page {page_num + 1} --- \n{text}")
            
            self.log.info("PDF read successfully", file=str(pdf_path), pages=len(all_text))
            return "\n".join(all_text)
            

        except Exception as e:
            self.log.error("Error reading PDF", file=str(pdf_path), error=str(e))
            raise DocumentPortalException("Error reading PDF", sys)
        
    def delete_existing_files(self):
        try:
            if self.base_dir.exists() and self.base_dir.is_dir():
                for file in self.base_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                        self.log.info("file deleted", path=str(file))
                self.log.info("Directory cleanedd", directory=str(self.base_dir))
        except Exception as e:
            self.log.error(f"Error deleting existing file: {e}")
            raise DocumentPortalException("An errror occurred while deleting existing file.", sys)
        

    def combine_documents(self)-> str:
        try:
            content_dict = {}
            doc_parts = []

            for filename in sorted(self.base_dir.iterdir()):
                if filename.is_file() and filename.suffix == ".pdf":
                    content_dict[filename.name] = self.read_pdf(filename)

            for filename, content in content_dict.items():
                doc_parts.append(f"Document: {filename}\n{content}")

            combined_text = "\n\n".join(doc_parts)
            self.log.info("Documents combined", count=len(doc_parts))
            return combined_text

        except Exception as e:
            self.log.error(f"Error Combining Documents: {e}")
            raise DocumentPortalException("An error occured while ombining documents.", sys)
    
