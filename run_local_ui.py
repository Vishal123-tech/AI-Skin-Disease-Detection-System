"""Start the local-only SkinScanix trial without opening a public sharing tunnel."""
import os

if __name__ == "__main__":
    os.environ["ACNE_FOCUS_MODE"] = "0"
    os.environ["ACNE_PRIORITY_MODE"] = "0"
    os.environ["GEMINI_API_KEY"] = ""
    from gradio_app import demo, UI_LAUNCH_KWARGS
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False,
                **UI_LAUNCH_KWARGS)
