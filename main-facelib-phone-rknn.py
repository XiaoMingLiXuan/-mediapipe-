import cv2
import mediapipe as mp
import numpy as np
import math
import time
import sys

# 导入 RKNN 运行时库
try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("错误: 无法导入 RKNNLite API。请确保已正确安装 RKNN Toolkit Lite Python包。")
    sys.exit(1)

# ==============================================================================
# 辅助函数
# --- MediaPipe 辅助函数 ---
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


# --- YOLO on RKNN 的处理函数 ---
def preprocess_for_yolo_rknn(image, model_width=640, model_height=640):

    # 使用 letterbox (或直接resize，但要与转换模型时一致)
    img_resized, _, _ = letterbox(image, new_shape=(model_height, model_width), auto=False)
    
    # BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    #  Batch 维度，从 HWC 变为 1HWC (NHWC)
    img_with_batch = np.expand_dims(img_rgb, axis=0)
    return img_with_batch

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup: r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto: dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill: dw, dh = 0.0, 0.0; new_unpad = (new_shape[1], new_shape[0]); ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad: im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def postprocess_for_yolo_rknn(outputs, image_shape, confidence_thresh=0.45, iou_thresh=0.5):

    # 检查 outputs 是否为 None，防止 TypeError
    if outputs is None:
        print("[POSTPROCESS_WARN] Received None as output, returning no boxes.")
        return []

    # 假设 outputs 是一个列表，包含3个输出张量，形状如 (1, N, 8)
    # 需要根据模型的实际输出来写这里的后处理逻辑
    
    # 这里的后处理逻辑需要与模型（YOLOv8）的输出格式相匹配
    # 这是一个简化的占位符，需要用真实的后处理逻辑替换
    # 例如：boxes, scores, class_ids = new_yolov8_post_process(outputs, conf_thresh, iou_thresh)
    final_boxes = [] # 这是一个占位符
    
    return final_boxes


# ==============================================================================
# 初始化和全局配置
# --- MediaPipe 初始化 ---
# --- MediaPipe 初始化 ---
print("Initializing MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
print("MediaPipe initialized.")

# --- RKNN 模型初始化 ---
print("Loading RKNN model for YOLO...")
rknn_lite = RKNNLite(verbose=False)
ret = rknn_lite.load_rknn('./yolov8n.rknn')
if ret != 0:
    print(f'Load RKNN model failed! Error code: {ret}')
    rknn_lite.release()
    sys.exit(ret)

print("Initializing RKNN runtime...")
ret = rknn_lite.init_runtime()  # 移除 core_mask 参数
if ret != 0:
    print(f'Init RKNN runtime failed! Error code: {ret}')
    rknn_lite.release()
    sys.exit(ret)
print("RKNN model loaded and initialized successfully.")


# --- 其他参数和计时器 ---
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_ROBUST_INDICES = {"v_pair1": (82, 312), "v_pair2": (13, 14), "v_pair3": (87, 317), "h_pair": (61, 291)}
EAR_THRESHOLD, MAR_THRESHOLD, YAW_THRESHOLD = 0.25, 0.3, 35
ALARM_DURATION_SECONDS, NO_FACE_DURATION_SECONDS, OCCLUSION_DURATION_SECONDS = 2.0, 3.0, 2.0
LOW_BRIGHTNESS_THRESHOLD, HIGH_BRIGHTNESS_THRESHOLD = 20, 240
NO_FACE_START_TIME, IS_SEATED = None, True
OCCLUSION_START_TIME, OCCLUSION_ALARM_TRIGGERED = None, False
EYE_EVENT_START_TIME, MOUTH_EVENT_START_TIME, HEAD_EVENT_START_TIME, PHONE_EVENT_START_TIME = None, None, None, None
EYE_ALARM_TRIGGERED, MOUTH_ALARM_TRIGGERED, HEAD_ALARM_TRIGGERED, PHONE_ALARM_TRIGGERED = False, False, False, False

# ==============================================================================
# 主循环
cap = cv2.VideoCapture(13)
if not cap.isOpened():
    print("错误: 无法打开摄像头。")
    rknn_lite.release()
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("摄像头已打开，开始处理视频流...")

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("警告: 无法抓取视频帧。")
        time.sleep(0.1)
        continue

    frame_count += 1
    image = cv2.flip(frame, 1) # 水平翻转图像
    
    # 创建一个副本用于绘制所有内容
    output_image = image.copy()

    # --- YOLO 手机检测 (NPU) ---
    yolo_input = preprocess_for_yolo_rknn(image)
    outputs = rknn_lite.inference(inputs=[yolo_input], data_format=['nhwc'])
    detected_boxes = postprocess_for_yolo_rknn(outputs, image.shape, confidence_thresh=0.5)
    phone_detected_this_frame = len(detected_boxes) > 0
    
    # 手机检测报警逻辑
    if phone_detected_this_frame:
        if PHONE_EVENT_START_TIME is None:
            PHONE_EVENT_START_TIME = time.time()
        elif time.time() - PHONE_EVENT_START_TIME > ALARM_DURATION_SECONDS:
            if not PHONE_ALARM_TRIGGERED:
                print("!!! PHONE ALARM TRIGGERED !!!")
                PHONE_ALARM_TRIGGERED = True
            cv2.putText(output_image, "ALARM: PHONE USE", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        PHONE_EVENT_START_TIME = None
        PHONE_ALARM_TRIGGERED = False

    # 在图像上绘制YOLO的检测框
    for box in detected_boxes:
        x, y, w, h = box # 假设 postprocess 返回 [x, y, w, h]
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(output_image, "Phone", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)


    # --- MediaPipe 人脸分析 (CPU) ---
    image_rgb_for_mp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb_for_mp.flags.writeable = False # 提高性能
    mp_results = face_mesh.process(image_rgb_for_mp)
    image_rgb_for_mp.flags.writeable = True # 恢复可写

    img_h, img_w, _ = image.shape

    if mp_results.multi_face_landmarks:
        NO_FACE_START_TIME = None # 重置无人脸计时器
        
        for face_landmarks in mp_results.multi_face_landmarks:
            # 绘制人脸关键点
            mp_drawing.draw_landmarks(
                image=output_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)
            
            # ---  核心计算逻辑  ---
            landmarks = face_landmarks.landmark
            left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
            right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)
            ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar_robust(landmarks, MOUTH_ROBUST_INDICES)

            # 头部姿态 (YAW) 计算
            # 选择6个点
        face_2d = np.array([
            (landmarks[33].x * img_w, landmarks[33].y * img_h),     # 左眼左角
            (landmarks[263].x * img_w, landmarks[263].y * img_h),    # 右眼右角
            (landmarks[1].x * img_w, landmarks[1].y * img_h),       # 鼻子顶端
            (landmarks[61].x * img_w, landmarks[61].y * img_h),     # 左嘴角
            (landmarks[291].x * img_w, landmarks[291].y * img_h),    # 右嘴角
            (landmarks[152].x * img_w, landmarks[152].y * img_h)     # 下巴最低点 (新增)
        ], dtype=np.float64)

        # 对应这6个点的通用3D模型坐标
        # 注意顺序必须与 face_2d 中的点一一对应
        face_3d = np.array([
            [-225.0, 170.0, -135.0],    # 左眼左角 (对应 face_2d[0])
            [225.0, 170.0, -135.0],     # 右眼右角 (对应 face_2d[1])
            [0.0, 0.0, 0.0],            # 鼻子顶端 (对应 face_2d[2])
            [-150.0, -150.0, -125.0],    # 左嘴角  (对应 face_2d[3])
            [150.0, -150.0, -125.0],     # 右嘴角  (对应 face_2d[4])
            [0.0, -330.0, -65.0]        # 下巴最低点 (对应 face_2d[5], 新增)
        ], dtype=np.float64)

        focal_length = img_w
        cam_matrix = np.array([[focal_length, 0, img_w / 2],
                               [0, focal_length, img_h / 2],
                               [0, 0, 1]])

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # solvePnP 有6个点
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        if success:
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            yaw_angle = angles[1] # YAW 角度


            # ---  核心报警逻辑  ---

            # 1. 眼睛闭合报警
            if ear < EAR_THRESHOLD:
                if EYE_EVENT_START_TIME is None:
                    EYE_EVENT_START_TIME = time.time()
                elif time.time() - EYE_EVENT_START_TIME > ALARM_DURATION_SECONDS:
                    if not EYE_ALARM_TRIGGERED:
                        print("!!! ALARM: EYES CLOSED FOR TOO LONG !!!")
                        EYE_ALARM_TRIGGERED = True
                    cv2.putText(output_image, "ALARM: EYES CLOSED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                EYE_EVENT_START_TIME = None
                EYE_ALARM_TRIGGERED = False

            # 2. 打哈欠报警
            if mar > MAR_THRESHOLD:
                if MOUTH_EVENT_START_TIME is None:
                    MOUTH_EVENT_START_TIME = time.time()
                elif time.time() - MOUTH_EVENT_START_TIME > ALARM_DURATION_SECONDS:
                    if not MOUTH_ALARM_TRIGGERED:
                        print("!!! ALARM: YAWNING DETECTED !!!")
                        MOUTH_ALARM_TRIGGERED = True
                    cv2.putText(output_image, "ALARM: YAWN", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                MOUTH_EVENT_START_TIME = None
                MOUTH_ALARM_TRIGGERED = False

            # 3. 头部姿态报警
            if abs(yaw_angle) > YAW_THRESHOLD:
                if HEAD_EVENT_START_TIME is None:
                    HEAD_EVENT_START_TIME = time.time()
                elif time.time() - HEAD_EVENT_START_TIME > ALARM_DURATION_SECONDS:
                    if not HEAD_ALARM_TRIGGERED:
                        print("!!! ALARM: HEAD TURNED AWAY !!!")
                        HEAD_ALARM_TRIGGERED = True
                    cv2.putText(output_image, "ALARM: DISTRACTED", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                HEAD_EVENT_START_TIME = None
                HEAD_ALARM_TRIGGERED = False
                
    else: # 如果没有检测到人脸
        if NO_FACE_START_TIME is None:
            NO_FACE_START_TIME = time.time()
        elif time.time() - NO_FACE_START_TIME > NO_FACE_DURATION_SECONDS:
            cv2.putText(output_image, "ALARM: NO FACE DETECTED", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # 你可以在这里添加无人脸的报警声音等
            
    # --- 添加调试文本，显示实时状态值 ---
    # if mp_results.multi_face_landmarks:
    #     cv2.putText(output_image, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    #     cv2.putText(output_image, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    #     cv2.putText(output_image, f"YAW: {yaw_angle:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


    # --- 显示图像 ---
    cv2.imshow('Fatigue Detection on RK3568', output_image)

    if cv2.waitKey(1) & 0xFF == 27: # 按 'ESC' 键退出
        break

# ==============================================================================
# 释放资源
print("Releasing resources...")
rknn_lite.release()  # 释放NPU资源
cap.release()
cv2.destroyAllWindows()
print("Done.")