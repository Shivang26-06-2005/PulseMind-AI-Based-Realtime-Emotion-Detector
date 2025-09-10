import cv2
import mediapipe as mp
import math
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize counters (only posture/gestures)
counters = {
    "Leaning Forward": 0,
    "Leaning Back": 0,
    "Leaning Left": 0,
    "Leaning Right": 0,
    "Upright Posture": 0,
    "Hand Raised": 0,
    "Hand Lowered": 0,
    "Pointing": 0,
    "Waving": 0,
}

baseline_z = None
cap = cv2.VideoCapture(0)

# Memory for wrist motion (waving)
wrist_history = {"left": deque(maxlen=5), "right": deque(maxlen=5)}

# History for movement detection (2 frames)
shoulder_history = deque(maxlen=2)
head_history = deque(maxlen=2)
left_wrist_history = deque(maxlen=2)
right_wrist_history = deque(maxlen=2)

def distance(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

def get_precise_direction(dx, dy, dz, thresh=0.01):
    """
    Returns a clear combined direction for 3D movement.
    Example outputs: "Forward-Right-Up"
    """
    directions = []
    if abs(dx) > thresh:
        directions.append("Right" if dx > 0 else "Left")
    if abs(dy) > thresh:
        directions.append("Down" if dy > 0 else "Up")
    if abs(dz) > thresh:
        directions.append("Back" if dz > 0 else "Forward")
    if not directions:
        directions.append("No Movement")
    return "-".join(directions)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        movement_info = []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # --- Torso midpoints for leaning detection ---
            l_sh, r_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_hip, r_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            mid_sh_x = (l_sh.x + r_sh.x)/2
            mid_sh_y = (l_sh.y + r_sh.y)/2
            mid_sh_z = (l_sh.z + r_sh.z)/2
            mid_hip_x = (l_hip.x + r_hip.x)/2
            mid_hip_y = (l_hip.y + r_hip.y)/2
            mid_hip_z = (l_hip.z + r_hip.z)/2

            if baseline_z is None:
                baseline_z = (mid_sh_z + mid_hip_z)/2

            dx = mid_hip_x - mid_sh_x
            dz = ((mid_hip_z + mid_sh_z)/2) - baseline_z
            dy_torso = mid_hip_y - mid_sh_y
            lean_thresh_z = 0.03
            lean_thresh_x = 0.03

            posture = "Upright Posture"
            if dz < -lean_thresh_z:
                if dx > lean_thresh_x:
                    posture = "Leaning Forward-Right"
                    counters["Leaning Forward"] +=1
                    counters["Leaning Right"] +=1
                elif dx < -lean_thresh_x:
                    posture = "Leaning Forward-Left"
                    counters["Leaning Forward"] +=1
                    counters["Leaning Left"] +=1
                else:
                    posture = "Leaning Forward"
                    counters["Leaning Forward"] +=1
            elif dz > lean_thresh_z:
                if dx > lean_thresh_x:
                    posture = "Leaning Back-Right"
                    counters["Leaning Back"] +=1
                    counters["Leaning Right"] +=1
                elif dx < -lean_thresh_x:
                    posture = "Leaning Back-Left"
                    counters["Leaning Back"] +=1
                    counters["Leaning Left"] +=1
                else:
                    posture = "Leaning Back"
                    counters["Leaning Back"] +=1
            else:
                if dx > lean_thresh_x:
                    posture = "Leaning Right"
                    counters["Leaning Right"] +=1
                elif dx < -lean_thresh_x:
                    posture = "Leaning Left"
                    counters["Leaning Left"] +=1

            cv2.putText(image, f"Posture: {posture}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            # --- Gestures ---
            l_elbow, r_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            l_wrist, r_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # Hand raised/lowered
            if l_wrist.y < l_sh.y or r_wrist.y < r_sh.y:
                counters["Hand Raised"] +=1
            else:
                counters["Hand Lowered"] +=1

            # Pointing
            l_angle = math.degrees(math.atan2(l_wrist.y - l_elbow.y, l_wrist.x - l_elbow.x))
            r_angle = math.degrees(math.atan2(r_wrist.y - r_elbow.y, r_wrist.x - r_elbow.x))
            if abs(l_angle)<30 or abs(r_angle)<30:
                counters["Pointing"] +=1

            # Waving
            wrist_history["left"].append(l_wrist.x)
            wrist_history["right"].append(r_wrist.x)
            if len(wrist_history["left"])==5 and max(wrist_history["left"]) - min(wrist_history["left"])>0.05:
                counters["Waving"] +=1
            if len(wrist_history["right"])==5 and max(wrist_history["right"]) - min(wrist_history["right"])>0.05:
                counters["Waving"] +=1

            # --- Movement detection (Blue Section) ---
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

            head_mid = ((nose.x + left_eye.x + right_eye.x)/3,
                        (nose.y + left_eye.y + right_eye.y)/3,
                        (nose.z + left_eye.z + right_eye.z)/3)
            head_history.append(head_mid)

            shoulder_mid = (mid_sh_x, mid_sh_y, mid_sh_z)
            shoulder_history.append(shoulder_mid)

            left_wrist_history.append((l_wrist.x, l_wrist.y, l_wrist.z))
            right_wrist_history.append((r_wrist.x, r_wrist.y, r_wrist.z))

            # Compute movement directions including Up/Down
            if len(shoulder_history)==2:
                dx, dy, dz = [shoulder_history[1][i]-shoulder_history[0][i] for i in range(3)]
                movement_info.append(f"Shoulders: {get_precise_direction(dx, dy, dz)}")

            if len(head_history)==2:
                dx, dy, dz = [head_history[1][i]-head_history[0][i] for i in range(3)]
                movement_info.append(f"Head/Neck/Eyes: {get_precise_direction(dx, dy, dz)}")

            if len(left_wrist_history)==2:
                dx, dy, dz = [left_wrist_history[1][i]-left_wrist_history[0][i] for i in range(3)]
                movement_info.append(f"Left Hand: {get_precise_direction(dx, dy, dz)}")

            if len(right_wrist_history)==2:
                dx, dy, dz = [right_wrist_history[1][i]-right_wrist_history[0][i] for i in range(3)]
                movement_info.append(f"Right Hand: {get_precise_direction(dx, dy, dz)}")

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --- Display counters (Green) ---
        start_y = 50
        for i, (k, v) in enumerate(counters.items()):
            cv2.putText(image, f"{k}: {v}", (10, start_y + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

        # --- Display movement info (Blue) ---
        start_y_mov = 50
        for i, info in enumerate(movement_info):
            cv2.putText(image, info, (400, start_y_mov + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)

        cv2.imshow('MediaPipe Gestures + Movement + Up/Down', image)
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
