import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.title("Live Moving Object Detection - YOLO11")
st.caption("Green = Static, Red = Moving. Running on limited CPU, so frames are processed every few ticks for smoothness.")

move_threshold = st.slider("Movement sensitivity (lower = more sensitive)", 1, 50, 8)
process_every_n = st.slider("Process every N frames (higher = smoother but less frequent detection)", 1, 10, 3)

model = YOLO("yolo11n.pt")

class MotionDetector(VideoProcessorBase):
    def __init__(self):
        self.prev_centers = {}
        self.move_threshold = 8
        self.process_every_n = 3
        self.frame_count = 0
        self.last_boxes = []  # cache last detections so freeze doesn't happen between processed frames

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Downscale for speed
        h0, w0 = img.shape[:2]
        scale = 640 / max(h0, w0)
        small = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))

        self.frame_count += 1
        if self.frame_count % self.process_every_n == 0:
            results = model.track(small, persist=True, conf=0.4, verbose=False)
            boxes_out = []
            if results[0].boxes.id is not None:
                for box, track_id in zip(results[0].boxes, results[0].boxes.id):
                    track_id = int(track_id)
                    x, y, w, h = box.xywh[0].tolist()
                    # scale back to original image size
                    x, y, w, h = x / scale, y / scale, w / scale, h / scale
                    cls_name = model.names[int(box.cls[0])]

                    dist = 0
                    status = "Static"
                    if track_id in self.prev_centers:
                        px, py = self.prev_centers[track_id]
                        dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                        if dist > self.move_threshold:
                            status = "Moving"
                    self.prev_centers[track_id] = (x, y)
                    boxes_out.append((x, y, w, h, cls_name, status, dist))
            self.last_boxes = boxes_out

        # Draw cached boxes every frame (so feed never looks frozen)
        for (x, y, w, h, cls_name, status, dist) in self.last_boxes:
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            color = (0, 0, 255) if status == "Moving" else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} - {status} ({dist:.1f}px)"
            cv2.putText(img, label, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=MotionDetector,
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
)

if ctx.video_processor:
    ctx.video_processor.move_threshold = move_threshold
    ctx.video_processor.process_every_n = process_every_n

