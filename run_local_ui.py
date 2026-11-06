"""Compatibility launcher; implementation lives in skinscanix/."""
import runpy

if __name__ == "__main__":
    runpy.run_module("skinscanix.run_local_ui", run_name="__main__")
