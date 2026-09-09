"""Optional private Google Drive storage for feedback images and metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path


SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _json_env(name: str) -> dict | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must contain valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{name} must contain a JSON object.")
    return parsed


def configured() -> bool:
    return bool(
        os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "").strip()
        and (
            os.environ.get("GOOGLE_DRIVE_TOKEN_JSON", "").strip()
            or (
                os.environ.get("GOOGLE_DRIVE_CLIENT_JSON", "").strip()
                and os.environ.get("GOOGLE_DRIVE_REFRESH_TOKEN", "").strip()
            )
        )
    )


def _credentials():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive support is not installed. Install requirements-google-drive.txt."
        ) from exc

    token_info = _json_env("GOOGLE_DRIVE_TOKEN_JSON")
    if token_info:
        credentials = Credentials.from_authorized_user_info(token_info, SCOPES)
    else:
        client_info = _json_env("GOOGLE_DRIVE_CLIENT_JSON")
        refresh_token = os.environ.get("GOOGLE_DRIVE_REFRESH_TOKEN", "").strip()
        if not client_info or not refresh_token:
            raise RuntimeError(
                "Set GOOGLE_DRIVE_TOKEN_JSON or GOOGLE_DRIVE_CLIENT_JSON plus "
                "GOOGLE_DRIVE_REFRESH_TOKEN."
            )
        details = client_info.get("installed") or client_info.get("web") or client_info
        credentials = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri=details.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=details.get("client_id"),
            client_secret=details.get("client_secret"),
            scopes=SCOPES,
        )

    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    return credentials


def upload_feedback(image_path: Path, record: dict) -> dict:
    """Upload one image and one private JSON record to the configured folder."""

    if not configured():
        return {"status": "disabled"}

    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload, MediaInMemoryUpload
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive support is not installed. Install requirements-google-drive.txt."
        ) from exc

    service = build("drive", "v3", credentials=_credentials(), cache_discovery=False)
    folder_id = os.environ["GOOGLE_DRIVE_FOLDER_ID"].strip()
    feedback_id = str(record["id"])
    suffix = image_path.suffix.lower() or ".jpg"

    image_file = service.files().create(
        body={
            "name": f"{feedback_id}{suffix}",
            "parents": [folder_id],
            "description": "Private feedback image for supervised model review",
        },
        media_body=MediaFileUpload(str(image_path), resumable=True),
        fields="id,name",
        supportsAllDrives=True,
    ).execute()

    metadata = dict(record)
    metadata["local_image_file"] = metadata.pop("image_file", "")
    metadata["drive_image_file_id"] = image_file.get("id", "")
    metadata_body = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
    metadata_file = service.files().create(
        body={
            "name": f"{feedback_id}.json",
            "parents": [folder_id],
            "description": "Private feedback metadata for supervised model review",
        },
        media_body=MediaInMemoryUpload(metadata_body, mimetype="application/json"),
        fields="id,name",
        supportsAllDrives=True,
    ).execute()

    return {
        "status": "uploaded",
        "image_file_id": image_file.get("id", ""),
        "metadata_file_id": metadata_file.get("id", ""),
    }
