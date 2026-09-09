"""Create a local Google Drive OAuth token for this project."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Google Drive OAuth token")
    parser.add_argument("client_json", type=Path, help="OAuth Desktop client JSON")
    parser.add_argument("--output", type=Path, default=Path("google_drive_token.json"))
    args = parser.parse_args()

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise SystemExit(
            "Install requirements-google-drive.txt before running this script."
        ) from exc

    flow = InstalledAppFlow.from_client_secrets_file(
        str(args.client_json), ["https://www.googleapis.com/auth/drive.file"]
    )
    credentials = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    args.output.write_text(credentials.to_json(), encoding="utf-8")
    print(f"Token created: {args.output.resolve()}")
    print("Keep this file private and store its JSON as GOOGLE_DRIVE_TOKEN_JSON in deployment secrets.")


if __name__ == "__main__":
    main()
