import cv2
import mediapipe as mp
import numpy as np
from fer import FER
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Initialize MediaPipe and FER
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
emotion_detector = FER(mtcnn=True)

# Emotion detection
def detect_emotion(frame):
    emotion, _ = emotion_detector.top_emotion(frame)
    return emotion if emotion else "Neutral"

# Posture detection
def detect_posture(landmarks):
    if landmarks:
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        if np.linalg.norm(left_shoulder - right_shoulder) < 0.2:
            return "Slouching"
        return "Upright"
    return "Unknown"

# Hand movement detection
def detect_hand_movement(landmarks):
    if landmarks:
        lw_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        rw_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        ls_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        rs_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        if lw_y < ls_y and rw_y < rs_y:
            return "Both Hands Up"
        elif lw_y < ls_y:
            return "Left Hand Up"
        elif rw_y < rs_y:
            return "Right Hand Up"
    return "Hands Down"

# Neck movement detection
def detect_neck_movement(landmarks):
    if landmarks:
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
        eye_y = landmarks[mp_pose.PoseLandmark.LEFT_EYE].y
        ls_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        rs_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        nose_x = landmarks[mp_pose.PoseLandmark.NOSE].x
        ls_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
        rs_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x

        mid_x = (ls_x + rs_x) / 2
        mid_y = (ls_y + rs_y) / 2
        tol_x = 0.05 * abs(ls_x - rs_x)

        horizontal = "Head Center"
        if nose_x < mid_x - tol_x:
            horizontal = "Head Left"
        elif nose_x > mid_x + tol_x:
            horizontal = "Head Right"

        vertical = "Head Level"
        if nose_y < eye_y:
            vertical = "Head Up"
        elif nose_y > mid_y:
            vertical = "Head Down"

        return f"{vertical}, {horizontal}"
    return "Unknown"

# Main video transformer
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.face = mp_face.FaceDetection(min_detection_confidence=0.7)
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Face and emotion
        face_result = self.face.process(rgb)
        emotion = detect_emotion(rgb) if face_result.detections else "Neutral"
        if face_result.detections:
            for det in face_result.detections:
                bbox = det.location_data.relative_bounding_box
                x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.putText(image, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 224), 2)

        # Pose detection
        pose_result = self.pose.process(rgb)
        if pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(image, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = pose_result.pose_landmarks.landmark
            posture = detect_posture(landmarks)
            hands_status = detect_hand_movement(landmarks)
            neck_status = detect_neck_movement(landmarks)

            cv2.putText(image, f"Posture: {posture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 224), 2)
            cv2.putText(image, f"Hand: {hands_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 224), 2)
            cv2.putText(image, f"Neck: {neck_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 224), 2)

        # Hand landmarks
        hand_result = self.hands.process(rgb)
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return image

# Streamlit UI
st.title("🧠 Real-Time Emotion, Pose & Gesture Detection")
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)