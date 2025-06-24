from abc import ABC, abstractmethod


class AbstractContext(ABC):
    def __init__(self, log=False):
        self._strategy = None
        self.log = log

    def __enter__(self):
        print(f"# [Pipeline] [{self.__class__.__name__}] started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log:
            print(f"# [Pipeline] [{self.__class__.__name__}] completed")

    def run(self):
        self._strategy = self._set_strategy()
        return self._execute_strategy()

    def _execute_strategy(self):
        if self._strategy is None:
            return None
        with self._strategy as strategy:
            return strategy.execute()

    @abstractmethod
    def _set_strategy(self):
        raise NotImplementedError("This method must be overwritten.")
