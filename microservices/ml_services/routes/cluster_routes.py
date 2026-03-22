# routes/cluster_routes.py
from fastapi import APIRouter, HTTPException
from schemas.cluster_schema import ClusterInput
from services.cluster_service import predict_cluster

router = APIRouter(prefix="/cluster", tags=["Cluster"])

# Cluster interpretation mapping
CLUSTER_LABELS = {
    0: {
        "label": "Low-Moderate Engagement",
        "description": "This post has low to moderate engagement. It's performing at an average or below-average level.",
        "recommendations": [
            "Consider posting at peak times",
            "Use more engaging visuals or headlines",
            "Add call-to-action prompts",
            "Engage with early commenters to boost visibility"
        ]
    },
    1: {
        "label": "High Engagement",
        "description": "This post is performing very well with high engagement! It's attracting significant attention and interactions.",
        "recommendations": [
            "Pin this post to keep it visible",
            "Respond to comments to maintain momentum",
            "Consider boosting or promoting similar content",
            "Analyze what made this successful for future posts"
        ]
    }
}

@router.post("/predict")
def predict(data: ClusterInput):
    """
    Predict the engagement cluster for a social media post.

    Returns the cluster classification with human-readable labels and recommendations.

    **Input Features:**
    - num_reactions: Total number of reactions
    - num_comments: Total number of comments
    - num_shares: Total number of shares
    - num_likes: Number of like reactions
    - num_loves: Number of love reactions
    - num_wows: Number of wow reactions
    - num_hahas: Number of haha reactions
    - num_sads: Number of sad reactions
    - num_angrys: Number of angry reactions
    """
    try:
        cluster_id = predict_cluster(data.dict())
        cluster_info = CLUSTER_LABELS.get(cluster_id, {
            "label": "Unknown",
            "description": "Cluster information not available",
            "recommendations": []
        })

        # Calculate total engagement
        total_engagement = (
            data.num_reactions +
            data.num_comments +
            data.num_shares
        )

        return {
            "cluster_id": cluster_id,
            "cluster_label": f"{cluster_info['label']}",
            "description": cluster_info['description'],
            "recommendations": cluster_info['recommendations'],
            "engagement_summary": {
                "total_reactions": data.num_reactions,
                "total_comments": data.num_comments,
                "total_shares": data.num_shares,
                "total_engagement": total_engagement,
                "sentiment_breakdown": {
                    "positive": data.num_likes + data.num_loves,
                    "neutral": data.num_wows + data.num_hahas,
                    "negative": data.num_sads + data.num_angrys
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))