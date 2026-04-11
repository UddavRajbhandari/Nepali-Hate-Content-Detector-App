"""
Prediction history endpoints.

GET    /api/history          — fetch saved predictions (paginated via limit/offset)
DELETE /api/history          — wipe entire history
GET    /api/history/stats    — summary counts and averages without fetching all rows
"""

from fastapi import APIRouter, HTTPException, Query
from backend.app.utils.history import load_history, clear_history

router = APIRouter()


@router.get(
    "/history",
    summary="Fetch prediction history",
    response_description="List of saved predictions, newest first",
)
async def get_history(
    limit: int = Query(default=100, ge=1, le=500, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Skip this many records from the newest end"),
):
    """
    Returns saved predictions in reverse-chronological order (newest first).
    Use limit + offset for basic pagination if history grows large.
    """
    history = load_history()
    total = len(history)

    reversed_history = list(reversed(history))
    page = reversed_history[offset : offset + limit]

    return {
        "items": page,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/history/stats",
    summary="Summary statistics over all saved predictions",
)
async def get_history_stats():
    """
    Returns aggregated stats without sending every record to the frontend.
    """
    history = load_history()

    if not history:
        return {
            "total": 0,
            "avg_confidence": None,
            "class_counts": {},
            "most_common_class": None,
        }

    total = len(history)
    avg_confidence = sum(h.get("confidence", 0.0) for h in history) / total

    class_counts: dict[str, int] = {}
    for h in history:
        label = h.get("prediction", "UNKNOWN")
        class_counts[label] = class_counts.get(label, 0) + 1

    most_common = max(class_counts, key=lambda k: class_counts[k])

    return {
        "total": total,
        "avg_confidence": round(avg_confidence, 4),
        "class_counts": class_counts,
        "most_common_class": most_common,
    }


@router.delete(
    "/history",
    summary="Clear all prediction history",
)
async def delete_history():
    """
    Permanently deletes the history file.
    Returns the number of records that were deleted for confirmation.
    """
    history = load_history()
    count = len(history)

    if count == 0:
        raise HTTPException(
            status_code=404,
            detail="History is already empty — nothing to clear.",
        )

    clear_history()

    return {
        "message": f"History cleared. {count} record(s) deleted.",
        "deleted_count": count,
    }
