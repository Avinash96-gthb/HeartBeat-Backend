from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import shutil
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from numpy.fft import fft, fftfreq

app = FastAPI()

# Enable CORS (update origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"],
)

@app.post("/estimate-heart-rate")
async def estimate_heart_rate(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video."}

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            return {"error": "Failed to get FPS from video."}

        peak_values_red = []
        peak_values_green = []
        peak_values_blue = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            red_channel = frame[:, :, 0]
            green_channel = frame[:, :, 1]
            blue_channel = frame[:, :, 2]

            peak_values_red.append(np.max(red_channel))
            peak_values_green.append(np.max(green_channel))
            peak_values_blue.append(np.max(blue_channel))

        cap.release()

        peak_values_red = np.array(peak_values_red)
        peak_values_green = np.array(peak_values_green)
        peak_values_blue = np.array(peak_values_blue)

        combined_peak_values = np.mean([peak_values_red, peak_values_green, peak_values_blue], axis=0)

        # Filtering
        def bandpass_filter(data, lowcut, highcut, fs, order=5):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band', analog=False)
            return filtfilt(b, a, data)

        filtered_combined_values = bandpass_filter(combined_peak_values, 0.8, 2.0, fps)

        threshold_amplitude = np.mean(filtered_combined_values) * 0.3
        peaks, _ = find_peaks(filtered_combined_values, height=threshold_amplitude)

        num_beats = len(peaks)
        if num_beats > 0:
            duration_in_seconds = len(filtered_combined_values) / fps
            duration_in_minutes = duration_in_seconds / 60
            heart_rate_bpm = num_beats / duration_in_minutes if duration_in_minutes > 0 else 0
        else:
            heart_rate_bpm = 0

        return {"heart_rate_bpm": round(heart_rate_bpm, 2)}

    finally:
        # Clean up temporary video
        os.remove(video_path)

# Run locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
