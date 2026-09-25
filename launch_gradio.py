"""Share launcher using the same interface and callbacks as the main app."""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
from gradio_app import demo, PORT, UI_LAUNCH_KWARGS


if __name__ == "__main__":
    print("=== LAUNCHING GRADIO ===", flush=True)
    _, local_url, share_url = demo.launch(
        server_name="0.0.0.0", server_port=PORT, share=True,
        prevent_thread_lock=True, **UI_LAUNCH_KWARGS,
    )
    print(f"LOCAL_URL={local_url}", flush=True)
    print(f"PUBLIC_SHARE_URL={share_url}", flush=True)
    if share_url:
        Path("public_url.txt").write_text(share_url, encoding="utf-8")
    demo.block_thread()
