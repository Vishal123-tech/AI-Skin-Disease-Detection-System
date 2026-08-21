import time
import requests
import cv2

# Server URL (Live Render endpoint)
SERVER_URL = "https://ai-skin-disease-detection-system.onrender.com/api/predict"

def capture_and_send():
    print("Capturing skin lesion image from camera...")
    cap = cv2.VideoCapture(0)
    time.sleep(1) # Give camera time to warm up/focus
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not capture image from camera.")
        return

    # Save temporary image
    temp_filename = "pi_lesion_capture.jpg"
    cv2.imwrite(temp_filename, frame)
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
