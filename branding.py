"""Shared SkinScanix branding; never modifies prediction or report values."""
import base64
from functools import lru_cache
from pathlib import Path

BRAND_NAME = "SkinScanix"
LOGO_PATH = Path(__file__).resolve().parent / "static" / "skinscanix-logo.png"


@lru_cache(maxsize=1)
def website_header() -> str:
    # Embed this one public asset; no private folders need to be exposed by Gradio.
    logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f'''<header aria-label="{BRAND_NAME}" style="display:flex;align-items:center;gap:18px;margin:8px 0 18px;">
      <img src="data:image/png;base64,{logo}" alt="{BRAND_NAME} logo" width="96" height="96"
        style="width:clamp(68px,14vw,96px);height:auto;flex-shrink:0;object-fit:contain;background:#fff;border-radius:16px;padding:5px;box-sizing:border-box;" />
      <div style="min-width:0;">
        <div style="font-size:14px;letter-spacing:normal;text-transform:none;font-weight:700;color:#57c9b6;margin-bottom:5px;">{BRAND_NAME}</div>
        <h1 style="font-size:clamp(21px,3vw,29px);font-weight:700;line-height:1.2;margin:0;color:inherit;">AI Skin Disease Detection</h1>
      </div>
    </header>'''
