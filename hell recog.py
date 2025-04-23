import cv2
import mediapipe as mp
import numpy as np
from fer import FER

# Initialize MediaPipe solutions and FER emotion detector
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
emotion_detector = FER(mtcnn=True)

# Set up the webcam and MediaPipe models
cap = cv2.VideoCapture(0)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Function to detect facial emotion using FER
def detect_emotion(frame):
    emotion, score = emotion_detector.top_emotion(frame)
    return emotion if emotion else "Neutral"


# Function to detect posture based on shoulder distance
def detect_posture(landmarks):
    if landmarks:
        left_shoulder = np.array(
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
        right_shoulder = np.array(
            [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_distance < 0.2:
            return "Slouching"
        return "Upright"
    return "Unknown"


# Function to detect hand movement based on wrist and shoulder landmarks
def detect_hand_movement(landmarks):
    if landmarks:
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        if left_wrist_y < left_shoulder_y and right_wrist_y < right_shoulder_y:
            return "Both Hands Up"
        elif left_wrist_y < left_shoulder_y:
            return "Right Hand Up"
        elif right_wrist_y < right_shoulder_y:
            return "Left Hand Up"
    return "Hands Down"


# Improved function to detect neck movement based on nose and shoulder positions
def detect_neck_movement(landmarks):
    if landmarks:
        # Get relevant landmarks
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
        left_eye_y = landmarks[mp_pose.PoseLandmark.LEFT_EYE].y
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        nose_x = landmarks[mp_pose.PoseLandmark.NOSE].x
        left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
        right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x

        # Calculate the midpoint between shoulders
        shoulder_mid_x = (left_shoulder_x + right_shoulder_x) / 2
        shoulder_mid_y = (left_shoulder_y + right_shoulder_y) / 2

        # Define a tolerance for left/right detection
        tolerance_x = 0.05 * abs(left_shoulder_x - right_shoulder_x)

        # Detect left/right head movement
        if abs(nose_x - shoulder_mid_x) <= tolerance_x:
            horizontal_movement = "Head Center"
        elif nose_x < shoulder_mid_x:
            horizontal_movement = "Head Left"
        else:
            horizontal_movement = "Head Right"

        # Detect up/down head movement
        if nose_y < left_eye_y:
            vertical_movement = "Head Up"
        elif nose_y > shoulder_mid_y:
            vertical_movement = "Head Down"
        else:
            vertical_movement = "Head Level"

        # Combine both movements
        return f"{vertical_movement}, {horizontal_movement}"

    return "Unknown"


# Main loop for capturing video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face and emotion
    face_results = face_detection.process(rgb_frame)
    emotion = detect_emotion(rgb_frame) if face_results.detections else "Neutral"
    for detection in face_results.detections if face_results.detections else []:
        bboxC = detection.location_data.relative_bounding_box
        bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
        cv2.rectangle(frame, bbox, (0, 0, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 224), 2)

    # Detect pose, posture, hand movement, and neck movement
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = pose_results.pose_landmarks.landmark
        posture = detect_posture(landmarks)
        hand_movement = detect_hand_movement(landmarks)
        neck_movement = detect_neck_movement(landmarks)

        # Display posture, hand, and neck status
        cv2.putText(frame, f"Posture: {posture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 224), 2)
        cv2.putText(frame, f"Hand Movement: {hand_movement}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 224), 2)
        cv2.putText(frame, f"Neck Movement: {neck_movement}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 224), 2)

    # Detect hands
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for landmark in hand_landmarks.landmark:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    # Display frame
    cv2.imshow("Face, Emotion, Posture, Hand & Neck Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
