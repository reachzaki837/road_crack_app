from flask import Flask, render_template, send_file, request, redirect, url_for
import cv2
import os
import time
import threading
import csv
from picamera2 import Picamera2
import numpy as np
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from collections import deque

app = Flask(__name__)

# Camera setup
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
def capture_from_camera():
    frame = picam2.capture_array()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Folders
image_folder = "static/results/images"
os.makedirs(image_folder, exist_ok=True)

UPLOAD_FOLDER = "static/results/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

CALIB_FOLDER = "static/results/calibration"
os.makedirs(CALIB_FOLDER, exist_ok=True)

# Load calibration points if exist
if os.path.exists("config.json"):
    with open("config.json", "r") as f:
        pts1 = np.float32(json.load(f))
else:
    pts1 = None


def correct_perspective(frame):
    global pts1
    if pts1 is None:
        return frame
    h, w = frame.shape[:2]
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(frame, matrix, (w, h))


# Log file
log_file = "static/results/road_log.csv"
if not os.path.isfile(log_file):  # don’t overwrite existing
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Image", "Crack %", "Potholes", "RHI", "Condition"])

# Global vars
latest_image, latest_rhi, latest_crack, latest_condition, latest_alert, latest_potholes = [None] * 6
capture_interval = 5
capture_enabled = False
recent_images = deque(maxlen=5)


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Crack detection
    crack_pixels = cv2.countNonZero(morph)
    total_pixels = morph.size
    crack_percent = (crack_pixels / total_pixels) * 100

    # Pothole detection
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pothole_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 800 < area < 10000:
            pothole_count += 1
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)

    # Road Health Index
    RHI = 100 - (crack_percent * 2) - (pothole_count * 5)
    RHI = max(0, min(100, RHI))

    if RHI >= 80:
        condition, alert = "Good", "green"
    elif RHI >= 60:
        condition, alert = "Moderate", "yellow"
    else:
        condition, alert = "Poor", "red"
               
    return frame, crack_percent, pothole_count, RHI, condition, alert


def capture_loop():
    global latest_image, latest_rhi, latest_crack, latest_condition, latest_alert, latest_potholes, recent_images
    while True:
        if capture_enabled:
            #success, frame = cap.read()
            try:
                frame = capture_from_camera()
            except Exception as e:
                print("Camera capture error:", e)
                continue
            frame = correct_perspective(frame)
            proc, crack_percent, potholes, RHI, condition, alert = process_frame(frame)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            img_name = f"{image_folder}/road_{timestamp}.jpg"
            cv2.imwrite(img_name, proc)

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), img_name, f"{crack_percent:.2f}", potholes, f"{RHI:.2f}", condition])

            latest_image, latest_rhi, latest_crack, latest_condition, latest_alert, latest_potholes = (
                img_name, RHI, crack_percent, condition, alert, potholes
            )

            recent_images.appendleft({"path": img_name, "rhi": round(RHI, 2), "cond": condition, "potholes": potholes})

            time.sleep(capture_interval)
        else:
            time.sleep(1)


threading.Thread(target=capture_loop, daemon=True).start()


@app.route("/", methods=["GET", "POST"])
def index():
    global capture_interval, capture_enabled
    if request.method == "POST":
        if "interval" in request.form:
            try:
                new_interval = int(request.form.get("interval"))
                if new_interval > 0:
                    capture_interval = new_interval
            except:
                pass
        elif "toggle" in request.form:
            capture_enabled = not capture_enabled
        return redirect(url_for("index"))

    return render_template("index.html", image=latest_image, crack=latest_crack, rhi=latest_rhi,
                           condition=latest_condition, alert=latest_alert, potholes=latest_potholes,
                           interval=capture_interval, enabled=capture_enabled, gallery=list(recent_images))


@app.route("/report")
def report():
    if os.path.isfile(log_file):
        try:
            df = pd.read_csv(log_file, on_bad_lines="skip")
            required_cols = ["Timestamp", "Image", "Crack %", "Potholes", "RHI", "Condition"]
            for col in required_cols:
                if col not in df.columns:
                    return "⚠️ Log format mismatch."

            df["RHI"] = pd.to_numeric(df["RHI"], errors="coerce")
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.dropna(subset=["RHI", "Timestamp"])

            if len(df) < 2:
                return "Not enough data."

            # Linear regression
            X = np.arange(len(df)).reshape(-1, 1)
            y = df["RHI"].values
            model = LinearRegression().fit(X, y)
            future_X = np.arange(len(df), len(df) + 5).reshape(-1, 1)
            predictions = model.predict(future_X)

            # ✅ Clamp predicted RHI and classify
            prediction_results = []
            for val in predictions:
                val = max(0, min(100, val))  # clamp
                if val >= 80:
                    cond = "Good"
                elif val >= 60:
                    cond = "Moderate"
                else:
                    cond = "Poor"
                prediction_results.append((round(val, 2), cond))

            # Plot observed + predicted
            plt.figure(figsize=(6, 4))
            plt.plot(df["Timestamp"], df["RHI"], marker="o", color="blue", label="Observed")
            last_time = df["Timestamp"].iloc[-1]
            future_times = pd.date_range(start=last_time, periods=6, freq="min")[1:]
            plt.plot(future_times, [p[0] for p in prediction_results],
                     marker="x", color="red", linestyle="--", label="Predicted")
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Time")
            plt.ylabel("RHI (0–100)")
            plt.title("Road Health Trend")
            plt.legend()
            plt.tight_layout()
            graph_path = "static/results/trend.png"
            plt.savefig(graph_path)
            plt.close()

            table_html = df.to_html(classes="table table-striped", index=False)
            return render_template("report.html", table=table_html,
                                   graph=graph_path, predictions=prediction_results)

        except Exception as e:
            return f"⚠️ Error: {str(e)}"
    else:
        return "No log yet."

@app.route("/download_log")
def download_log():
    return send_file(log_file, as_attachment=True)


@app.route("/upload", methods=["POST"])
def upload():
    global latest_image, latest_rhi, latest_crack, latest_condition, latest_alert, latest_potholes, recent_images
    if "file" not in request.files:
        return redirect(url_for("index"))
    files = request.files.getlist("file")
    for file in files:
        if file.filename == "":
            continue
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        frame = cv2.imread(filepath)
        if frame is None:
            continue

        frame = correct_perspective(frame)
        proc, crack_percent, potholes, RHI, condition, alert = process_frame(frame)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(image_folder, f"upload_{timestamp}.jpg")
        cv2.imwrite(save_path, proc)

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), save_path, f"{crack_percent:.2f}", potholes, f"{RHI:.2f}", condition])

        latest_image, latest_rhi, latest_crack, latest_condition, latest_alert, latest_potholes = (
            save_path, RHI, crack_percent, condition, alert, potholes
        )

        recent_images.appendleft({"path": save_path, "rhi": round(RHI, 2), "cond": condition, "potholes": potholes})
    return redirect(url_for("report"))


@app.route("/calibrate")
def calibrate():
    if latest_image:
        return render_template("calibrate.html", image=latest_image)
    else:
        files = os.listdir(CALIB_FOLDER)
        if files:
            return render_template("calibrate.html", image=os.path.join(CALIB_FOLDER, files[-1]))
        return "⚠️ No image for calibration."


@app.route("/save_points", methods=["POST"])
def save_points():
    global pts1
    data = request.get_json()
    pts1 = np.float32(data["points"])
    with open("config.json", "w") as f:
        json.dump(data["points"], f)
    return {"status": "success", "message": "Calibration saved"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)

import atexit
@atexit.register
def cleanup_camera():
    try:
        picam2.stop()
        print("Camera stopped successfully.")
    except:
        pass



'''
#Pi-Camera Setup Instructions
# 1. Install Picamera2 library
sudo apt update
sudo apt install python3-picamera2
# 2. Modify the code as follows:
from picamera2 import Picamera2
import cv2
# Camera setup for Raspberry Pi Camera Module using Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

def get_frame():
    frame = picam2.capture_array()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#Replace this line: 
cap = cv2.VideoCapture(0)
#With this line:
def capture_from_camera():
    return get_frame()
#in your capture_loop, instead of:
success, frame = cap.read()
#Use:
frame = capture_from_camera()

# Alternative Camera Setup for USB Camera on Raspberry Pi
# If using a USB camera, use the following setup:
# Camera setup for USB Camera on Raspberry Pi

picam2 = Picamera2()
picam2.start()
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise Exception("Could not open video device")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 10)
'''
