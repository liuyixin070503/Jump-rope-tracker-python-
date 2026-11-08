import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


model_path = "pose_landmarker_lite.task"

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
)

landmarker = vision.PoseLandmarker.create_from_options(options)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("摄像头打开失败，请检查摄像头权限。")
else:
    print("姿态预览已启动。按 Q 键退出。")

    frame_timestamp_ms = 0
    jump_count = 0
    start_time = None
    elapsed_seconds = 0
    is_running = False
    is_airborne = False
    previous_hip_y = None

    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


    jump_up_threshold = -0.44
    jump_down_min = -0.39
    jump_down_max = -0.35

    while True:
        success, frame = camera.read()

        if not success:
            print("读取摄像头画面失败。")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        relative_height = None
        hip_movement = None

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            key_points = [
                landmarks[LEFT_HIP],
                landmarks[RIGHT_HIP],
                landmarks[LEFT_ANKLE],
                landmarks[RIGHT_ANKLE],
            ]

            if all(point.visibility > 0.1 for point in key_points):
                hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
                if previous_hip_y is None:
                    hip_movement = 0
                else:
                    hip_movement = hip_y - previous_hip_y

                previous_hip_y = hip_y
                ankle_y = (
                    landmarks[LEFT_ANKLE].y
                    + landmarks[RIGHT_ANKLE].y
                ) / 2
                relative_height = hip_y - ankle_y

                if is_running:
                    if relative_height < jump_up_threshold:
                        is_airborne = True

                    elif (
                            is_airborne
                            and jump_down_min <= relative_height <= jump_down_max
                    ):
                        jump_count += 1
                        is_airborne = False
            height, width, _ = frame.shape

            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

            for connection in connections:
                start = landmarks[connection.start]
                end = landmarks[connection.end]

                start_point = (int(start.x * width), int(start.y * height))
                end_point = (int(end.x * width), int(end.y * height))

                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        if is_running:
            elapsed_seconds = int(time.time() - start_time)

        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60
        timer_text = f"{minutes:02d}:{seconds:02d}"
        # 简洁信息面板
        panel_width = 300
        cv2.rectangle(frame, (12, 12), (panel_width, 142), (25, 25, 25), -1)
        cv2.rectangle(frame, (12, 12), (panel_width, 142), (80, 200, 120), 2)

        if is_running:
            status_text = "RUNNING"
            status_color = (80, 220, 120)
        else:
            status_text = "READY  |  Press S to start"
            status_color = (0, 200, 255)

        cv2.putText(
            frame,
            status_text,
            (26, 42),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            status_color,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"JUMPS   {jump_count}",
            (26, 82),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"TIME      {timer_text}",
            (26, 116),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "S Start/Restart    E Finish    Q Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        cv2.imshow("跳绳计数与计时", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            jump_count = 0
            elapsed_seconds = 0
            start_time = time.time()
            is_running = True
            is_airborne = False
            print("运动开始")

        elif key == ord("e"):
            if is_running:
                elapsed_seconds = int(time.time() - start_time)
                is_running = False
                is_airborne = False
                print("运动结束")

        elif key == ord("q"):
            break


camera.release()
landmarker.close()
cv2.destroyAllWindows()
