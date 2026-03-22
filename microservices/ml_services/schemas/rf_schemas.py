from pydantic import BaseModel

class RFInput(BaseModel):
    buying: str
    maint: str
    doors: str
    persons: str
    lug_boot: str
    safety: str