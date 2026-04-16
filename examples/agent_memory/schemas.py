from pydantic import BaseModel


class RandomWordOutput(BaseModel):
    random_word: str
