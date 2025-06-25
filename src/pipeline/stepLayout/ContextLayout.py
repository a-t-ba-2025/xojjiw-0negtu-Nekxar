from ..AbstractContext import AbstractContext
from src.pipeline.stepLayout.postprocessor.LayoutPostprocessor import LayoutPostprocessor
from .layoutStrategy.StrategyFRCNN import StrategyFRCNN
from .layoutStrategy.StrategyDETR import StrategyDETR
from .layoutStrategy.StrategyHybridFRCNN_DETR import StrategyHybridFRCNN_DETR
from .layoutStrategy.StrategyPDF import StrategyPDF


class ContextLayout(AbstractContext):
    def __init__(self, file_type, text_json, words=None, image=None, pdf_path=None, log=False):
        super().__init__(log)
        self.words = words
        self.file_type = file_type
        self.image = image
        self.pdf_path = pdf_path
        self.text_json = text_json

    def run(self):
        self._strategy = self._set_strategy()
        if self.file_type == "pdf":
            image_with_boxes, box_results = self._execute_strategy()
            return image_with_boxes, box_results
        else:
            image_with_boxes, box_results = self._execute_strategy()
            post_processor = LayoutPostprocessor(text_json=self.text_json, log=self.log)
            layout = post_processor.run(box_results)
            return image_with_boxes, layout

    def _set_strategy(self):
        if self.file_type == "pdf":
            return StrategyPDF(pdf_path=self.pdf_path, words=self.words, log=self.log)
        return StrategyFRCNN(image=self.image, log=self.log)
        # return StrategyDETR(image=self.image, log=self.log)  # not used
        # return StrategyHybridFRCNN_DETR(image=self.image, log=self.log) # not used
