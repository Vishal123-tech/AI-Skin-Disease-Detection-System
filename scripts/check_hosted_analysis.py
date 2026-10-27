"""Exercise uploads/queue/PDF with synthetic images, never patient photos."""
import argparse
from pathlib import Path
from time import perf_counter
from PIL import Image
from gradio_client import Client, handle_file

parser = argparse.ArgumentParser()
parser.add_argument("url")
args = parser.parse_args()
folder = Path("work/local_trial")
folder.mkdir(parents=True, exist_ok=True)
photo = folder / "synthetic_48mp.jpg"
if not photo.exists():
    Image.new("RGB", (8000, 6000), (180, 130, 100)).save(photo)
client = Client(args.url, verbose=False)
started = perf_counter()
jobs = [client.submit(handle_file(str(photo)), api_name="/analyze") for _ in range(3)]
for number, job in enumerate(jobs, 1):
    result = job.result(timeout=120)
    assert result[1] and Path(result[1]).is_file(), result
    print(f"request {number}: PDF received, elapsed={perf_counter()-started:.2f}s", flush=True)
print("Three uploads completed successfully.")
