from pydantic import BaseModel, Field
from typing import List, Optional

#=============DTOs==================
class CandidateOptionDTO(BaseModel):
    id: str = Field(..., description="Unique key to identify the option (e.g., 'opt_1')")
    text: str = Field(..., max_length=200, description="The English action description")
    category: Optional[str] = Field(None, description="Optional tag (e.g., 'emergency', 'maintenance')")

class OptionMatrixRequestDTO(BaseModel):
    query_id: str = Field(..., description="Correlates these options to the user prompt")
    options: List[CandidateOptionDTO] = Field(
        ..., 
        min_length=2, 
        max_length=32, 
        description="List of candidate actions to score (minimum 2, maximum 32)"
    )
#===================================


