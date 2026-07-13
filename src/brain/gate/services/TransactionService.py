from typing import Callable, List
from ..interfaces.TransactionManager import TransactionManager
from ..models.WriteRequest import WriteRequest

class TransactionService:

    def __init__(self, managers: List[TransactionManager]) -> None:
        self.managers = managers

    def execute_transaction(self, request: WriteRequest, operation: Callable[[], None]) -> None:
        self.validate_request(request)
        for manager in self.managers:
            manager.begin()

        try:
            operation()
            for manager in self.managers:
                manager.commit()
        except Exception as e:
            for manager in self.managers:
                try:
                    manager.rollback()
                except Exception:
                    pass
            raise e

    def validate_request(self, request: WriteRequest) -> None:
        if not request:
            raise ValueError("WriteRequest cannot be None")
        if not request.action:
            raise ValueError("WriteRequest action must be specified")
        
        valid_actions = {"SAVE", "UPDATE", "DELETE", "CLEAR"}
        if request.action not in valid_actions:
            raise ValueError(f"Unknown action: {request.action}. Must be one of {valid_actions}")
        
        if request.action in {"SAVE", "UPDATE", "DELETE"} and not request.entry_id:
            raise ValueError(f"entry_id is required for action {request.action}")
