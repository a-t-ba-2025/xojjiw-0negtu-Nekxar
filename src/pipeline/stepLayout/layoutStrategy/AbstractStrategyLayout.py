from abc import ABC, abstractmethod


class AbstractStrategyLayout(ABC):
    def __init__(self, image, log: bool = False):
        self.image = image
        self.log = log

    def __enter__(self):
        print(f"## [Pipeline] [ContextLayout] [{self.__class__.__name__}] started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log:
            print(f"## [Pipeline][ContextLayout] [{self.__class__.__name__}] completed")

    @abstractmethod
    def execute(self):
        raise NotImplementedError("This method must be overwritten.")
