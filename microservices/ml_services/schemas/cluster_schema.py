# schemas/cluster_schema.py
from pydantic import BaseModel

class ClusterInput(BaseModel):
    num_reactions: int
    num_comments: int
    num_shares: int
    num_likes: int
    num_loves: int
    num_wows: int
    num_hahas: int
    num_sads: int
    num_angrys: int