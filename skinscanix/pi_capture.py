import os
import sys
import time
import requests
import cv2

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Use the Raspberry Pi's local Flask server by default. Override this when
# intentionally sending to a hosted API: $env:SERVER_URL=... or export SERVER_URL=...
SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:5000/api/predict")


def capture_image(filename: str) -> bool:
    """Capture with Picamera2 when available, otherwise use OpenCV."""
    try:
        from picamera2 import Picamera2

        camera = Picamera2()
        camera.configure(camera.create_still_configuration(main={"size": (1280, 960)}))
        camera.start()
        time.sleep(2)
        camera.capture_file(filename)
        camera.stop()
        return True
    except ImportError:
        print("Picamera2 is not installed; trying OpenCV camera capture.")
    except Exception as exc:
        print(f"Picamera2 capture failed: {exc}")

    cap = cv2.VideoCapture(0)
    time.sleep(1)
    ret, frame = cap.read()
    cap.release()
    return bool(ret and cv2.imwrite(filename, frame))

def capture_and_send():
    print("Capturing skin lesion image from camera...")
    temp_filename = "pi_lesion_capture.jpg"
    if not capture_image(temp_filename):
        print("Error: Could not capture image from camera.")
        return

    print(f"Image saved locally as {temp_filename}")

    # Send image to the website API
    print(f"Uploading image to {SERVER_URL}...")
    with open(temp_filename, "rb") as img_file:
        files = {"image": (temp_filename, img_file, "image/jpeg")}
        response = requests.post(SERVER_URL, files=files)

    if response.status_code == 200:
        data = response.json()
        print("\n--- ANALYSIS RESULTS ---")
        print(f"Prediction : {data.get('label')}")
        print(f"Confidence : {data.get('confidence')}")
        print(f"Quality    : {data.get('quality')}")
        print(f"PDF Report : {data.get('report_url')}")
    else:
        print(f"Error ({response.status_code}):", response.text)

if __name__ == "__main__":
    capture_and_send()
