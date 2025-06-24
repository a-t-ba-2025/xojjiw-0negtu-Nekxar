from ..AbstractContext import AbstractContext
from src.pipeline.stepContent.contentStrategy.StrategyContentPipeline import StrategyContentPipeline


class ContextContent(AbstractContext):
    def __init__(self, image, text_json, layout_json, log=False):
        super().__init__(log)
        self.image = image
        self.text_json = text_json
        self.layout_json = layout_json

    def _set_strategy(self):
        return StrategyContentPipeline(
            image=self.image,
            text_json=self.text_json,
            layout_json=self.layout_json,
            log=self.log
        )
