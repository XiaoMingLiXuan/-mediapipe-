import cv2
import mediapipe as mp
import numpy as np
import math
import time

# --- 辅助函数 ---
def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def calculate_ear(landmarks, eye_indices):
    p2_p6 = euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    p3_p5 = euclidean_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    p1_p4 = euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    if p1_p4 == 0: return 0.0
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def calculate_mar_robust(landmarks, mouth_indices):
    v_dist1 = euclidean_distance(landmarks[mouth_indices["v_pair1"][0]], landmarks[mouth_indices["v_pair1"][1]])
    v_dist2 = euclidean_distance(landmarks[mouth_indices["v_pair2"][0]], landmarks[mouth_indices["v_pair2"][1]])
    v_dist3 = euclidean_distance(landmarks[mouth_indices["v_pair3"][0]], landmarks[mouth_indices["v_pair3"][1]])
    avg_vertical_dist = (v_dist1 + v_dist2 + v_dist3) / 3
    horizontal_dist = euclidean_distance(landmarks[mouth_indices["h_pair"][0]], landmarks[mouth_indices["h_pair"][1]])
    if horizontal_dist == 0: return 0.0
    return avg_vertical_dist / horizontal_dist

# --- 初始化 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_ROBUST_INDICES = {"v_pair1": (82, 312), "v_pair2": (13, 14), "v_pair3": (87, 317), "h_pair": (61, 291)}

# --- 可调参数 ---
EAR_THRESHOLD, MAR_THRESHOLD, YAW_THRESHOLD = 0.25, 0.3, 35
ALARM_DURATION_SECONDS, NO_FACE_DURATION_SECONDS, OCCLUSION_DURATION_SECONDS = 2.0, 3.0, 2.0
LOW_BRIGHTNESS_THRESHOLD, HIGH_BRIGHTNESS_THRESHOLD = 20, 240

# --- 状态和计时器初始化 ---
NO_FACE_START_TIME, IS_SEATED = None, True
OCCLUSION_START_TIME, OCCLUSION_ALARM_TRIGGERED = None, False
EYE_EVENT_START_TIME, MOUTH_EVENT_START_TIME, HEAD_EVENT_START_TIME = None, None, None
EYE_ALARM_TRIGGERED, MOUTH_ALARM_TRIGGERED, HEAD_ALARM_TRIGGERED = False, False, False

# --- 主循环 ---
cap = cv2.VideoCapture(13)

# ========== 1. 设置摄像头分辨率和帧率 ==========
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 10)
# =======================================================

# ========== 2. 用于计算和显示FPS的变量 ==========
fps_start_time = 0
frame_count = 0
fps = 0
# =======================================================

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # --- FPS 计算逻辑 ---
    frame_count += 1
    # 每秒更新一次FPS值
    if (time.time() - fps_start_time) > 1:
        fps = frame_count / (time.time() - fps_start_time)
        fps_start_time = time.time()
        frame_count = 0

    image = cv2.flip(frame, 1)
    
    # 摄像头遮挡检测
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_image)
    if avg_brightness < LOW_BRIGHTNESS_THRESHOLD or avg_brightness > HIGH_BRIGHTNESS_THRESHOLD:
        if OCCLUSION_START_TIME is None: OCCLUSION_START_TIME = time.time()
        elif time.time() - OCCLUSION_START_TIME > OCCLUSION_DURATION_SECONDS: OCCLUSION_ALARM_TRIGGERED = True
    else:
        OCCLUSION_START_TIME, OCCLUSION_ALARM_TRIGGERED = None, False
    
    if OCCLUSION_ALARM_TRIGGERED:
        cv2.putText(image, "ALARM: CAMERA OCCLUDED!", (50, image.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow('Fatigue and Distraction Detection', image)
        if cv2.waitKey(5) & 0xFF == 27: break
        continue

    # 人脸分析
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image_h, image_w, _ = image.shape
    avg_ear_val, mar_val, yaw_val = 0.0, 0.0, 0.0
    
    # 状态判断
    if results.multi_face_landmarks:
        IS_SEATED = True
        NO_FACE_START_TIME = None
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            # 闭眼
            avg_ear_val = (calculate_ear(landmarks, LEFT_EYE_INDICES) + calculate_ear(landmarks, RIGHT_EYE_INDICES)) / 2.0
            if avg_ear_val < EAR_THRESHOLD:
                if EYE_EVENT_START_TIME is None: EYE_EVENT_START_TIME = time.time()
                elif time.time() - EYE_EVENT_START_TIME > ALARM_DURATION_SECONDS: EYE_ALARM_TRIGGERED = True
            else: EYE_EVENT_START_TIME, EYE_ALARM_TRIGGERED = None, False
            # 打哈欠
            mar_val = calculate_mar_robust(landmarks, MOUTH_ROBUST_INDICES)
            if mar_val > MAR_THRESHOLD:
                if MOUTH_EVENT_START_TIME is None: MOUTH_EVENT_START_TIME = time.time()
                elif time.time() - MOUTH_EVENT_START_TIME > ALARM_DURATION_SECONDS: MOUTH_ALARM_TRIGGERED = True
            else: MOUTH_EVENT_START_TIME, MOUTH_ALARM_TRIGGERED = None, False
            # 转头
            face_2d, model_points_3d = [], np.array([(0.0,0.0,0.0), (0.0,-330.0,-65.0), (-225.0,170.0,-135.0), (225.0,170.0,-135.0),(-150.0,-150.0,-125.0), (150.0,-150.0,-125.0)])
            for idx in [1, 199, 33, 263, 61, 291]: face_2d.append([int(landmarks[idx].x * image_w), int(landmarks[idx].y * image_h)])
            face_2d = np.array(face_2d, dtype=np.float64)
            focal_length, cam_matrix = 1*image_w, np.array([[1*image_w,0,image_h/2], [0,1*image_w,image_w/2], [0,0,1]])
            success, rot_vec, _ = cv2.solvePnP(model_points_3d, face_2d, cam_matrix, np.zeros((4,1)))
            if success:
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
                yaw_val = angles[1]
                if abs(yaw_val) > YAW_THRESHOLD:
                    if HEAD_EVENT_START_TIME is None: HEAD_EVENT_START_TIME = time.time()
                    elif time.time() - HEAD_EVENT_START_TIME > ALARM_DURATION_SECONDS: HEAD_ALARM_TRIGGERED = True
                else: HEAD_EVENT_START_TIME, HEAD_ALARM_TRIGGERED = None, False
            mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=drawing_spec)
    else:
        if NO_FACE_START_TIME is None: NO_FACE_START_TIME = time.time()
        elif time.time() - NO_FACE_START_TIME > NO_FACE_DURATION_SECONDS: IS_SEATED = False
        EYE_ALARM_TRIGGERED, MOUTH_ALARM_TRIGGERED, HEAD_ALARM_TRIGGERED = False, False, False

    # --- 显示逻辑 ---
    # 就座状态
    presence_text = "Status: Seated" if IS_SEATED else "Status: Away"
    presence_color = (0, 255, 0) if IS_SEATED else (0, 165, 255)
    cv2.putText(image, presence_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, presence_color, 2)
    
    # 疲劳/分心状态
    if IS_SEATED:
        eye_text = "ALARM: EYES CLOSED!" if EYE_ALARM_TRIGGERED else "Eye: Normal"
        eye_color = (0, 0, 255) if EYE_ALARM_TRIGGERED else (255, 255, 255)
        cv2.putText(image, eye_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2)
        mouth_text = "ALARM: YAWNING!" if MOUTH_ALARM_TRIGGERED else "Mouth: Normal"
        mouth_color = (0, 0, 255) if MOUTH_ALARM_TRIGGERED else (255, 255, 255)
        cv2.putText(image, mouth_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mouth_color, 2)
        head_text = "ALARM: HEAD TURN!" if HEAD_ALARM_TRIGGERED else "Head: Normal"
        head_color = (0, 0, 255) if HEAD_ALARM_TRIGGERED else (255, 255, 255)
        cv2.putText(image, head_text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, head_color, 2)
    
    # 调试数值
    cv2.putText(image, f"EAR: {avg_ear_val:.2f}", (image_w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"MAR: {mar_val:.2f}", (image_w - 150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Yaw: {yaw_val:.2f}", (image_w - 150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Bright: {avg_brightness:.0f}", (image_w - 150, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ========== 3. 在屏幕上显示FPS ==========
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(image, fps_text, (image_w - 150, image_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # =======================================================

    cv2.imshow('Fatigue and Distraction Detection', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()