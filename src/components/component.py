from abc import ABC, abstractmethod
import pandas as pd

class Component(ABC):

    """Abstract base class for components."""
    @abstractmethod
    def load_data(self, filepath: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_data(self, df: pd.DataFrame, filepath: str) -> None:
        pass
