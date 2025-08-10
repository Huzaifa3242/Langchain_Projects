from typing import Annotated
from pydantic import BaseModel,Field
class output(BaseModel):
    score :float = Field(gt=0,le=10,description="score of the cv")
    summary :str = Field(description="Summary of the cv")
    improvement:str = Field(description="To tell the user about improvement")
    keywords:Annotated[list[str],Field(description="list of the keywords extracted")]