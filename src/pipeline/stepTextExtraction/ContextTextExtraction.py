from ..AbstractContext import AbstractContext
from .textExtractionStrategy.StrategyPaddle import StrategyPaddle
from .textExtractionStrategy.StrategyPdf import StrategyPdf
from .textExtractionStrategy.StrategyTesseract import StrategyTesseract


class ContextTextExtraction(AbstractContext):
    def __init__(self, file_type, image, is_mostly_text, pdf_path=None, log=False):
        super().__init__(log)
        self.file_type = file_type
        self.image = image
        self.is_mostly_text = is_mostly_text
        self.pdf_path = pdf_path

    def _set_strategy(self):
        if self.file_type == "pdf":
            return StrategyPdf(pdf_path=self.pdf_path, log=self.log)
        elif self.is_mostly_text:
            return StrategyTesseract(image=self.image, log=self.log)
        else:
            return StrategyPaddle(image=self.image, log=self.log)
