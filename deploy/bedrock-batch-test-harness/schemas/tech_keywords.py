from pydantic import BaseModel, Field


class TechKeywords(BaseModel):
    computer_languages: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    other: list[str] = Field(default_factory=list)

class RandomName(BaseModel):
    name: str
