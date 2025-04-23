import cv2
import mediapipe as mp
import numpy as np
import time
from fer import FER
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, AudioProcessorBase
from collections import deque
import cx_Oracle  # For Oracle DB integration

# Oracle DB Configuration - Replace with your credentials
DB_USER = "ARC"
DB_PASSWORD = "Arc@2004"
DB_DSN = "localhost:1521/XE"

# Initialize MediaPipe and FER
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_mesh = mp.solutions.face_mesh
emotion_detector = FER(mtcnn=True)

cap = cv2.VideoCapture(0)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Helpers
def detect_emotion(frame):
    emotion, _ = emotion_detector.top_emotion(frame)
    return emotion if emotion else "Neutral"

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

def detect_hand_movement(landmarks):
    if landmarks:
        tolerance = 0.05  # New: Small buffer to allow slight hand raise detection

        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        # Adjust conditions with tolerance
        if left_wrist_y < (left_shoulder_y + tolerance) and right_wrist_y < (right_shoulder_y + tolerance):
            return "Both Hands Up"
        elif left_wrist_y < (left_shoulder_y + tolerance):
            return "Left Hand Up"
        elif right_wrist_y < (right_shoulder_y + tolerance):
            return "Right Hand Up"
    return "Hands Down"

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
        tol_x = 0.1 * abs(ls_x - rs_x)
        horizontal = "Head Center"
        if nose_x < mid_x - tol_x:
            horizontal = "Head Right"
        elif nose_x > mid_x + tol_x:
            horizontal = "Head Left"
        vertical = "Head Level"
        if nose_y < eye_y:
            vertical = "Head Up"
        elif nose_y > mid_y:
            vertical = "Head Down"
        return f"{vertical}, {horizontal}"
    return "Unknown"

def is_attentive(emotion, posture, hands_status, neck_status, gaze_forward, yawning):
    attentive_emotions = ["neutral", "happy", "surprise"]
    return (emotion.lower() in attentive_emotions and
            posture == "Upright" and
            hands_status in ["Hands Down", "Right Hand Up", "Left Hand Up"] and
            neck_status == "Head Level, Head Center" and
            gaze_forward and not yawning)

def is_looking_forward(landmarks):
    try:
        left_eye_ids = [33, 133]
        right_eye_ids = [362, 263]
        iris_left_id = 468
        iris_right_id = 473
        left = landmarks[left_eye_ids[0]]
        right = landmarks[right_eye_ids[1]]
        iris_l = landmarks[iris_left_id]
        iris_r = landmarks[iris_right_id]
        iris_center = (iris_l.x + iris_r.x) / 2
        eye_center = (left.x + right.x) / 2
        ratio = abs(iris_center - eye_center)
        return ratio < 0.07
    except:
        return False

def detect_yawn(landmarks):
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    return abs(upper_lip.y - lower_lip.y) > 0.05

# Oracle DB helpers
def save_score_to_db(score):
    try:
        conn = cx_Oracle.connect(DB_USER, DB_PASSWORD, DB_DSN)
        cur = conn.cursor()
        cur.execute("INSERT INTO attention_scores (score) VALUES (:1)", [score])
        conn.commit()
        cur.close()
        conn.close()
    except cx_Oracle.DatabaseError as e:
        st.error(f"Database error occurred: {e}")

def get_latest_score_from_db():
    try:
        conn = cx_Oracle.connect(DB_USER, DB_PASSWORD, DB_DSN)
        cur = conn.cursor()
        cur.execute("SELECT score FROM (SELECT score FROM attention_scores ORDER BY timestamp DESC) WHERE ROWNUM = 1")
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else 0
    except cx_Oracle.DatabaseError as e:
        st.error(f"Database error occurred: {e}")
        return 0

# Streamlit App
st.title("🧠 Real-Time Student Attention Detection")

# Shared attention score variable
st.session_state.attention_score = 0

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.face = mp_face.FaceDetection(min_detection_confidence=0.7)
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mesh = mp_mesh.FaceMesh(refine_landmarks=True)
        self.attention_history = deque(maxlen=60)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_result = self.face.process(rgb)
        face_mesh_result = self.mesh.process(rgb)
        emotion = detect_emotion(rgb) if face_result.detections else "Neutral"

        gaze_forward, yawning = False, False

        if face_result.detections:
            for det in face_result.detections:
                bbox = det.location_data.relative_bounding_box
                x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.putText(image, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 224), 2)

        if face_mesh_result.multi_face_landmarks:
            mesh_landmarks = face_mesh_result.multi_face_landmarks[0].landmark
            gaze_forward = is_looking_forward(mesh_landmarks)
            yawning = detect_yawn(mesh_landmarks)
            cv2.putText(image, f"Gaze: {'Looking Forward' if gaze_forward else 'Looking Away'}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 224), 2)
            if yawning:
                cv2.putText(image, "Yawning Detected!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        pose_result = self.pose.process(rgb)
        posture, hands_status, neck_status = "Unknown", "Hands Down", "Unknown"
        if pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(image, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = pose_result.pose_landmarks.landmark
            posture = detect_posture(landmarks)
            hands_status = detect_hand_movement(landmarks)
            neck_status = detect_neck_movement(landmarks)
            cv2.putText(image, f"Posture: {posture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 224), 2)
            cv2.putText(image, f"Hand: {hands_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 224), 2)
            cv2.putText(image, f"Neck: {neck_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 224), 2)

        attentive = is_attentive(emotion, posture, hands_status, neck_status, gaze_forward, yawning)
        self.attention_history.append(attentive)
        attention_score = int(100 * sum(self.attention_history) / len(self.attention_history)) if self.attention_history else 0
        st.session_state.attention_score = attention_score

        bar_x, bar_y, bar_w, bar_h = 10, 210, 230, 20
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        fill_w = int(bar_w * attention_score / 100)
        bar_color = (0, 255, 0) if attention_score >= 75 else (0, 255, 255) if attention_score >= 50 else (0, 0, 255)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        cv2.putText(image, f"Attention Score: {attention_score}%", (bar_x, bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 255), 2)

        return image


ctx = webrtc_streamer(key="video", video_processor_factory=VideoProcessor)

if ctx and not ctx.state.playing:
    # Store final score when video stops playing
    final_score = st.session_state.get("attention_score", 0)
    st.session_state["final_score"] = final_score
    print(f"Final Attention Score: {final_score}")  # Debug print to verify the score
    save_score_to_db(final_score)  # Save score to DB

if st.button("Show Final Attention Score"):
    latest_score = get_latest_score_from_db()
    print(f"Latest Score: {latest_score}")  # Debug print to check the latest score
    st.success(f"🎯 Final Attention Score: {latest_score}%")