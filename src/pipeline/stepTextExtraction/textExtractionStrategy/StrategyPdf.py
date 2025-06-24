import pdfplumber
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import red
import io

from src.pipeline.stepTextExtraction.textExtractionStrategy.AbstractStrategyTextExtraction import AbstractStrategyTextExtraction


class StrategyPdf(AbstractStrategyTextExtraction):
    def __init__(self, pdf_path, log: bool = False):
        super().__init__(log)
        self.pdf_path = pdf_path

    def execute(self):
        if self.log:
            print(f"### Extracting text from PDF: {self.pdf_path}")

        reader = PdfReader(self.pdf_path)  # Open PDF
        writer = PdfWriter()  # Create PDF writer for annotated files
        result = []  # List for store extracted text and positions

        # Open the PDF with pdfplumber
        with pdfplumber.open(self.pdf_path) as pdf:
            page = pdf.pages[0]  # Only first page
            # Extract all words with position
            words = page.extract_words(keep_blank_chars=True, use_text_flow=True, extra_attrs=["size", "fontname"])

            mem_file = io.BytesIO()  # Memory file for overlay drawing
            c = canvas.Canvas(mem_file, pagesize=(page.width, page.height))  # PDF canvas

            for word in words:
                text = word["text"]
                if not text.strip():
                    continue  # Skip empty entries

                # Bounding box
                x0 = word["x0"]
                x1 = word["x1"]
                top = page.height - word["top"]
                bottom = page.height - word["bottom"]
                width = x1 - x0
                height = top - bottom

                # Draw box
                c.setStrokeColor(red)
                c.setLineWidth(0.5)
                c.rect(x0, bottom, width, height, stroke=1, fill=0)

                # Save result
                result.append({
                    "text": text,
                    "bbox": [x0, bottom, x1, top],
                    "confidence": 1.0
                })

            c.save()  # Finish drawing
            mem_file.seek(0)  # beginning of  memory file
            overlay_pdf = PdfReader(mem_file)  # Read drawn  file as PDF

            page_ob = reader.pages[0] # Merge with original page
            page_ob.merge_page(overlay_pdf.pages[0])  # Overlay drawing
            writer.add_page(page_ob)  # Add annotated to the output

        memory_pdf = io.BytesIO()  # Create  inmemory file final PDF
        writer.write(memory_pdf)  # Write annotated page into it
        memory_pdf.seek(0)  # Reset position

        if self.log:
            print(f"### Annotated PDF created in memory")

        return memory_pdf, result, words