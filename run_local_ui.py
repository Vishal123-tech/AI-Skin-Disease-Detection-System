"""Start the local SkinScanix trial UI.

Supports Wi-Fi / mobile access via --host 0.0.0.0 and optional public
Gradio share tunnel via --share.
"""
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkinScanix Local UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind (0.0.0.0 = all interfaces for Wi-Fi/mobile access)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Generate a public Gradio share link (useful when Wi-Fi isolation is active)",
    )
    args = parser.parse_args()

    os.environ["ACNE_FOCUS_MODE"] = "0"
    os.environ["ACNE_PRIORITY_MODE"] = "0"
    os.environ["GEMINI_API_KEY"] = ""

    from gradio_app import demo, UI_LAUNCH_KWARGS

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        **UI_LAUNCH_KWARGS,
    )
