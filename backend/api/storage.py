import json
import os
from datetime import datetime
from typing import Dict, List, Any
import uuid


def ensure_dir(directory: str):
    """Ensure directory exists"""
    os.makedirs(directory, exist_ok=True)


def create_meeting_session(room_name: str, data_dir: str) -> Dict[str, Any]:
    """Create a new meeting session"""
    meeting_id = str(uuid.uuid4())
    session = {
        "meeting_id": meeting_id,
        "room_name": room_name,
        "started_at": datetime.now().isoformat(),
        "transcripts": [],
        "nudges": []
    }
    return session


def save_meeting_session(session: Dict[str, Any], data_dir: str):
    """Save meeting session to JSON file"""
    ensure_dir(os.path.join(data_dir, "meetings"))
    file_path = os.path.join(data_dir, "meetings", f"{session['meeting_id']}.json")
    with open(file_path, 'w') as f:
        json.dump(session, f, indent=2)


def add_transcript(session: Dict[str, Any], timestamp: str, speaker: str, text: str, is_final: bool):
    """Add transcript to session"""
    transcript = {
        "timestamp": timestamp,
        "speaker": speaker,
        "text": text,
        "is_final": is_final
    }
    session["transcripts"].append(transcript)
    return transcript


def add_nudge(session: Dict[str, Any], nudge_type: str, title: str, quote: str, confidence: float):
    """Add nudge to session"""
    nudge = {
        "type": nudge_type,
        "title": title,
        "quote": quote,
        "confidence": confidence,
        "created_at": datetime.now().isoformat()
    }
    session["nudges"].append(nudge)
    return nudge


def load_meeting_session(meeting_id: str, data_dir: str) -> Dict[str, Any]:
    """Load meeting session from JSON file"""
    file_path = os.path.join(data_dir, "meetings", f"{meeting_id}.json")
    with open(file_path, 'r') as f:
        return json.load(f)


def get_recent_transcripts(session: Dict[str, Any], count: int = 10) -> List[Dict[str, Any]]:
    """Get most recent final transcripts"""
    final_transcripts = [t for t in session["transcripts"] if t["is_final"]]
    return final_transcripts[-count:]
