from abc import ABC, abstractmethod

class TransactionManager(ABC):
    """Abstract interface representing transaction boundary coordination."""

    @abstractmethod
    def begin(self) -> None:
        """Begin a database transaction."""
        pass

    @abstractmethod
    def commit(self) -> None:
        """Commit the current database transaction."""
        pass

    @abstractmethod
    def rollback(self) -> None:
        """Rollback the current database transaction."""
        pass
