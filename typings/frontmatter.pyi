from typing import Any

class Post:
    metadata: dict[str, Any]
    content: str

def load(path: str) -> Post: ...

def loads(text: str) -> Post: ...
