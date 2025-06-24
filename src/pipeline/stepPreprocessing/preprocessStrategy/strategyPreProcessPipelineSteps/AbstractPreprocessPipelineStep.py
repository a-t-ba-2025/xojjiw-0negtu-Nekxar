from abc import ABC, abstractmethod


class AbstractPreprocessPipelineStep(ABC):
    def __init__(self, image, log: bool = False):
        self.log = log
        self.image = image

    def __enter__(self):
        print(f"### [Pipeline] [ContextPreprocessor] [StrategyPreprocesspipeline] [{self.__class__.__name__}] started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log:
            print(f"### [Pipeline] [ContextPreprocessor] [StrategyPreprocesspipeline] [{self.__class__.__name__}] completed")

    @abstractmethod
    def apply(self):
        raise NotImplementedError("This method must be overwritten.")
