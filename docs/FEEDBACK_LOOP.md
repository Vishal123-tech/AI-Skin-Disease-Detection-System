# Feedback and supervised improvement

The Gradio interface now shows a feedback section after each analysis:

1. Choose **Correct**, **Wrong**, or **Not sure**.
2. For a wrong result, select the correct condition and optionally add a note.
3. Click **Save Feedback**.

Feedback is written locally to `feedback/feedback.jsonl`, and the submitted
image is copied to `feedback/images/` with a random identifier. The folder is
ignored by Git so images are not pushed to GitHub, Hugging Face, or Render.

The interface requires consent before storing an image. The current save path
is local only; the optional Google Drive helpers are not connected to it.

One-click feedback training is paused following a regression: the active
four-class checkpoint had been trained with one feedback image and no replay
data, and predicted Eczema on every image in a 79-image local validation split.
The status button no longer trains or overwrites weights. Previously stored
images, weights, and training history are preserved.

This is a collection step, not automatic learning. Before retraining, review
the records, remove incorrect labels, confirm consent and licensing, and keep
patient-identifying information out of the images. Then create a train/val/test
dataset from the verified records, retrain in Colab, compare the new model with
the current model, and deploy only an approved model artifact.

Render's normal filesystem is not a permanent feedback database. Google Drive
is acceptable for this small private prototype, but for a larger public
deployment use a secure database/object-storage service and restrict access to
reviewed feedback.
