"""Compatibility launcher; implementation lives in skinscanix/."""
import runpy

if __name__ == "__main__":
    runpy.run_module("skinscanix.gradio_app", run_name="__main__")
