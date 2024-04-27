from pydantic import BaseModel


class InpaintModel(BaseModel):
    sketch: str
    model: str
    