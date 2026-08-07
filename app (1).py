import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.title("Live Moving Object Detection - YOLO11")

model = YOLO("yolo11n.pt")

class MotionDetector(VideoProcessorBase):
    def __init__(self):
        self.prev_centers = {}  # track_id -> last (x, y)
        self.move_threshold = 15  # pixels — tune this if detection feels too sensitive/insensitive

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.track(img, persist=True, conf=0.4, verbose=False)

        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes, results[0].boxes.id):
                track_id = int(track_id)
                x, y, w, h = box.xywh[0].tolist()
                cls_name = model.names[int(box.cls[0])]

                status = "Static"
                if track_id in self.prev_centers:
                    px, py = self.prev_centers[track_id]
                    dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                    if dist > self.move_threshold:
                        status = "Moving"

                self.prev_centers[track_id] = (x, y)

                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                color = (0, 0, 255) if status == "Moving" else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{cls_name} - {status}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="detection", video_processor_factory=MotionDetector)
