from __future__ import annotations

import os
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Initialize MongoDB connection globally
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ai_image_project")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
users = db["users"]
prediction_history = db["prediction_history"]
explanation_feedback = db["explanation_feedback"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def save_prediction_history(record: dict) -> str | None:
    """Persist prediction history item. Returns inserted id string if successful."""
    try:
        payload = dict(record)
        payload.setdefault("timestamp", utcnow())
        payload.setdefault("user_key", "anonymous")
        inserted = prediction_history.insert_one(payload)
        return str(inserted.inserted_id)
    except PyMongoError:
        return None


def fetch_recent_history(limit: int = 12, user_key: str | None = None) -> list[dict]:
    """Fetch recent prediction history records in descending timestamp order."""
    try:
        query = {"user_key": user_key} if user_key else {}
        rows = list(prediction_history.find(query).sort("timestamp", -1).limit(limit))
        for row in rows:
            row["_id"] = str(row["_id"])
        return rows
    except PyMongoError:
        return []


def delete_prediction_history(record_id: str, user_key: str | None = None) -> bool:
    """Delete one prediction history record by id."""
    try:
        try:
            query = {"_id": ObjectId(record_id)}
        except (InvalidId, TypeError, ValueError):
            query = {"_id": record_id}
        if user_key:
            query["user_key"] = user_key
        deleted = prediction_history.delete_one(query)
        return deleted.deleted_count > 0
    except PyMongoError:
        return False


def save_feedback(record: dict) -> str | None:
    """Persist user feedback for explanation quality."""
    try:
        payload = dict(record)
        payload.setdefault("timestamp", utcnow())
        inserted = explanation_feedback.insert_one(payload)
        return str(inserted.inserted_id)
    except PyMongoError:
        return None
