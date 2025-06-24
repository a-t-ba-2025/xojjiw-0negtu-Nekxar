from abc import ABC, abstractmethod


class AbstractContentPipelineStep(ABC):
    def __init__(self, image, text_json, layout_json, log: bool = False):
        self.log = log
        self.image = image
        self.text_json = text_json
        self.layout_json = layout_json

    def __enter__(self):
        print(f"### [Pipeline] [ContextContent] [StrategyContentPipeline] [{self.__class__.__name__}] started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log:
            print(f"### [Pipeline] [ContextContent] [StrategyContentPipeline] [{self.__class__.__name__}] completed")

    @abstractmethod
    def apply(self):
        raise NotImplementedError("This method must be overwritten.")
