import tkinter as tk
import cv2
from PIL import Image, ImageTk
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import sys
from tkinter import messagebox

camera = None
camera_running = False
workout_running = False
workout_start_time = None
elapsed_seconds = 0
if getattr(sys, "frozen", False):
    app_folder = sys._MEIPASS
else:
    app_folder = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(app_folder, "pose_landmarker_lite.task")
jump_count = 0
is_airborne = False
workout_paused = False
pose_ready = False

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

jump_up_threshold = -0.42
jump_down_min = -0.39
jump_down_max = -0.36

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
)

landmarker = vision.PoseLandmarker.create_from_options(pose_options)
frame_timestamp_ms = 0


def open_camera():
    global camera, camera_running

    if camera_running:
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        status_label.config(
            text="状态：摄像头打开失败，请检查权限或关闭占用摄像头的软件",
            fg="red"
        )
        camera.release()
        camera = None
        return

    camera_running = True
    camera_button.config(state="disabled")
    close_camera_button.config(state="normal")
    start_button.config(state="normal")
    status_label.config(text="状态：摄像头已打开，请调整位置让全身入镜", fg="green")
    update_camera()

def update_camera():
        global frame_timestamp_ms, jump_count, is_airborne, pose_ready
        if not camera_running or camera is None:
            return

        success, frame = camera.read()
        pose_ready = False

        if success:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            frame_timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                key_points = [
                    landmarks[LEFT_HIP],
                    landmarks[RIGHT_HIP],
                    landmarks[LEFT_ANKLE],
                    landmarks[RIGHT_ANKLE],
                ]

                pose_ready = all(
                    point.visibility > 0.1 for point in key_points
                )

                if workout_running:
                    if not pose_ready:
                        status_label.config(
                            text="状态：未识别到完整姿态，请调整位置；当前暂停计数",
                            fg="orange"
                        )
                    elif workout_paused:
                        status_label.config(text="状态：已暂停", fg="orange")
                    else:
                        status_label.config(text="状态：运动中", fg="green")
                else:
                    if pose_ready:
                        status_label.config(
                            text="状态：已识别到全身，可以开始运动",
                            fg="green"
                        )
                    else:
                        status_label.config(
                            text="状态：请调整位置，让髋部和双脚都进入画面",
                            fg="orange"
                        )

                if workout_running and not workout_paused and all(
                        point.visibility > 0.1 for point in key_points
                ):
                    hip_y = (
                                    landmarks[LEFT_HIP].y
                                    + landmarks[RIGHT_HIP].y
                            ) / 2

                    ankle_y = (
                                      landmarks[LEFT_ANKLE].y
                                      + landmarks[RIGHT_ANKLE].y
                              ) / 2

                    relative_height = hip_y - ankle_y

                    if relative_height < jump_up_threshold:
                        is_airborne = True

                    elif (
                            is_airborne
                            and jump_down_min <= relative_height <= jump_down_max
                    ):
                        jump_count += 1
                        is_airborne = False
                        count_label.config(text=f"跳跃次数：{jump_count}")

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
            else:
                if workout_running and not workout_paused:
                    status_label.config(
                        text="状态：未检测到人体，请到摄像头前",
                        fg="orange"
                    )

            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(display_frame)
            image = image.resize((640, 360))

            photo = ImageTk.PhotoImage(image=image)
            video_placeholder.config(image=photo, text="")
            video_placeholder.image = photo
        window.after(30, update_camera)
def close_camera():
    global camera, camera_running, workout_running

    if workout_running:
        status_label.config(text="状态：请先结束运动，再关闭摄像头", fg="red")
        return

    camera_running = False
    camera_button.config(state="normal")
    close_camera_button.config(state="disabled")
    start_button.config(state="disabled")
    workout_running = False

    if camera is not None:
        camera.release()
        camera = None

    video_placeholder.config(
        image="",
        text="摄像头画面将在这里显示"
    )
    video_placeholder.image = None
    status_label.config(text="状态：摄像头已关闭", fg="blue")

def start_workout():
    global workout_running, workout_paused
    global workout_start_time, elapsed_seconds
    global jump_count, is_airborne

    if not camera_running:
        status_label.config(text="状态：请先打开摄像头", fg="red")
        return
    if not pose_ready:
        status_label.config(
            text="状态：请进入画面，确保髋部和双脚都能看见",
            fg="orange"
        )
        return

    workout_running = True
    start_button.config(state="disabled")
    pause_button.config(state="normal")
    finish_button.config(state="normal")
    workout_paused = False
    elapsed_seconds = 0
    workout_start_time = time.time()
    jump_count = 0
    is_airborne = False

    count_label.config(text="跳跃次数：0")
    status_label.config(text="状态：运动中", fg="green")
    pause_button.config(text="暂停")
    show_timer()
    update_timer()


def toggle_pause():
    global workout_paused, workout_start_time, elapsed_seconds

    if not workout_running:
        status_label.config(text="状态：请先开始运动", fg="red")
        return

    if not workout_paused:
        elapsed_seconds += int(time.time() - workout_start_time)
        workout_paused = True
        status_label.config(text="状态：已暂停", fg="orange")
        pause_button.config(text="继续")
        show_timer()
    else:
        workout_start_time = time.time()
        workout_paused = False
        status_label.config(text="状态：运动中", fg="green")
        pause_button.config(text="暂停")
        update_timer()


def finish_workout():
    global workout_running, workout_paused, elapsed_seconds

    if workout_running and not workout_paused:
        elapsed_seconds += int(time.time() - workout_start_time)

    final_count = jump_count
    final_seconds = elapsed_seconds

    workout_running = False
    workout_paused = False
    elapsed_seconds = 0

    pause_button.config(text="暂停")
    status_label.config(text="状态：运动已结束，计时已归零", fg="blue")
    show_timer()
    start_button.config(state="normal" if camera_running else "disabled")
    pause_button.config(state="disabled")
    finish_button.config(state="disabled")

    minutes = final_seconds // 60
    seconds = final_seconds % 60
    messagebox.showinfo(
        "本次运动结果",
        f"跳跃次数：{final_count}\n运动时长：{minutes:02d}:{seconds:02d}"
    )

def show_timer():
    minutes = elapsed_seconds // 60
    seconds = elapsed_seconds % 60
    timer_label.config(text=f"运动时间：{minutes:02d}:{seconds:02d}")


def update_timer():
    if workout_running and not workout_paused:
        current_elapsed = elapsed_seconds + int(time.time() - workout_start_time)
        minutes = current_elapsed // 60
        seconds = current_elapsed % 60
        timer_label.config(text=f"运动时间：{minutes:02d}:{seconds:02d}")
        window.after(200, update_timer)
def close_app():
    if workout_running:
        status_label.config(
            text="状态：请先结束运动，再退出程序",
            fg="red"
        )
        return

    close_camera()
    landmarker.close()
    window.destroy()

window = tk.Tk()
window.title("跳绳计数与计时")
window.geometry("900x700")
window.configure(bg="#F4F7FB")

header = tk.Frame(window, bg="#142B4A", height=92)
header.pack(fill="x")
header.pack_propagate(False)

header_canvas = tk.Canvas(
    header,
    bg="#142B4A",
    height=92,
    highlightthickness=0
)
header_canvas.pack(fill="both", expand=True)

header_canvas.create_oval(825, 12, 870, 57, fill="#1E426B", outline="")
header_canvas.create_oval(847, 34, 865, 52, fill="#55E6C1", outline="")
header_canvas.create_oval(790, 65, 800, 75, fill="#FFCA6A", outline="")

header_canvas.create_text(
    33, 32,
    text="跳绳计数与计时",
    anchor="w",
    font=("Microsoft YaHei UI", 25, "bold"),
    fill="#07182B"
)

header_canvas.create_text(
    30, 29,
    text="跳绳计数与计时",
    anchor="w",
    font=("Microsoft YaHei UI", 25, "bold"),
    fill="#FFFFFF"
)

header_canvas.create_text(
    32, 66,
    text="MOVE  •  JUMP  •  KEEP GOING",
    anchor="w",
    font=("Bahnschrift SemiBold", 10),
    fill="#55E6C1"
)

header_canvas.create_line(
    30, 82, 178, 82,
    fill="#FFCA6A",
    width=3,
    capstyle="round"
)

instruction_label = tk.Label(
    window,
    text="先打开摄像头，调整位置让全身入镜，再开始运动。",
    font=("Microsoft YaHei UI", 11),
    bg="#f2f4f7",
    fg="#555555"
)
instruction_label.pack(pady=(0, 12))

video_area = tk.Frame(
    window,
    width=640,
    height=360,
    bg="#20252b",
    highlightthickness=1,
    highlightbackground="#c8cdd3"
)
video_area.pack()
video_area.pack_propagate(False)

video_placeholder = tk.Label(
    video_area,
    text="摄像头画面将在这里显示",
    font=("Microsoft YaHei UI", 15),
    bg="#20252b",
    fg="white"
)
video_placeholder.pack(expand=True)

stats_frame = tk.Frame(window, bg="#f2f4f7")
stats_frame.pack(pady=16)

count_label = tk.Label(
    stats_frame,
    text="跳跃次数：0",
    font=("Microsoft YaHei UI", 16, "bold"),
    bg="#f2f4f7"
)
count_label.pack(side="left", padx=30)

timer_label = tk.Label(
    stats_frame,
    text="运动时间：00:00",
    font=("Microsoft YaHei UI", 16, "bold"),
    bg="#f2f4f7"
)
timer_label.pack(side="left", padx=30)

# 操作按钮
buttons_frame = tk.Frame(window, bg="#f2f4f7")
buttons_frame.pack(pady=8)

camera_button = tk.Button(
    buttons_frame,
    text="打开摄像头",
    width=14,
    command=open_camera
)
camera_button.pack(side="left", padx=8)

close_camera_button = tk.Button(
    buttons_frame,
    state="disabled",
    text="关闭摄像头",
    width=14,
    command=close_camera
)
close_camera_button.pack(side="left", padx=8)

start_button = tk.Button(
    buttons_frame,
    text="开始运动",
    width=14,
    command=start_workout,
    state="disabled"
)
start_button.pack(side="left", padx=8)

pause_button = tk.Button(
    buttons_frame,
    state="disabled",
    text="暂停",
    width=14,
    command=toggle_pause
)
pause_button.pack(side="left", padx=8)

finish_button = tk.Button(
    buttons_frame,
    text="结束运动",
    width=14,
    command=finish_workout,
    state="disabled"
)
finish_button.pack(side="left", padx=8)

status_label = tk.Label(
    window,
    text="状态：等待操作",
    font=("Microsoft YaHei UI", 11),
    bg="#f2f4f7",
    fg="blue"
)
status_label.pack(pady=12)

window.protocol("WM_DELETE_WINDOW", close_app)
window.mainloop()
