from ..AbstractContext import AbstractContext
from .preprocessStrategy.StrategyPreProcessPipeline import StrategyPreProcessPipeline


class ContextPreprocessor(AbstractContext):
    def __init__(self, file_type, typed_file, log=False):
        super().__init__(log)
        self.file_type = file_type
        self.typed_file = typed_file

    def _set_strategy(self):
        if self.file_type == "pdf":
            print("## [Pipeline] [ContextPreprocessor] No preprocessing necessary in textbased PDFs")
            return None
        return StrategyPreProcessPipeline(image=self.typed_file, log=self.log)
