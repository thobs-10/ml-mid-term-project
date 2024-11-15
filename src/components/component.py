from abc import ABC, abstractmethod

class Component(ABC):
    """Abstract base class for components."""
    @abstractmethod
    def load_data(self) -> None:
        pass

    @abstractmethod
    def split_data(self) -> None:
        pass

    @abstractmethod
    def save_data(self) -> None:
        pass