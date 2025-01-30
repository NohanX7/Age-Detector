import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from ttkbootstrap import Style
import threading

# Load model OpenCV
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-5)', '(6-14)', '(15-22)', '(23-27)', '(28-38)', '(39-50)', '(50-80)', '(80-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

padding = 20

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frameOpencvDnn, faceBoxes

def detect_age_gender(image_path=None, camera_index=0):
    video = None
    if image_path:
        frame = cv2.imread(image_path)
        process_frame(frame, image_path)
    else:
        video = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not video.isOpened():
            messagebox.showerror("Error", f"Webcam dengan index {camera_index} tidak ditemukan!")
            return

        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            process_frame(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        video.release()
    cv2.destroyAllWindows()

def process_frame(frame, image_path=None):
    if frame is None:
        messagebox.showerror("Error", "Gagal membaca gambar!")
        return

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
                     max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Detecting age and gender", resultImg)

    if image_path:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        thread = threading.Thread(target=detect_age_gender, args=(file_path,))
        thread.start()

def open_webcam():
    camera_index = simpledialog.askinteger("Select Camera", "Masukkan indeks kamera (default: 0):", minvalue=0)
    if camera_index is None:
        return
    thread = threading.Thread(target=detect_age_gender, args=(None, camera_index))
    thread.start()

def show_help():
    messagebox.showinfo("Help", "Cara menggunakan:\n\n"
                                "- Klik 'Browse Image' untuk memilih gambar.\n"
                                "- Klik 'Open Webcam' untuk mendeteksi via kamera.\n"
                                "- Tekan 'q' atau ESC untuk menutup kamera.")

def quit_app():
    root.quit()
    root.destroy()

# GUI Setup
root = tk.Tk()
root.title("Age & Gender Detector")

# Menggunakan tema modern dengan ttkbootstrap
style = Style(theme="darkly")

# Center the window
window_width = 400
window_height = 350
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = int((screen_width / 2) - (window_width / 2))
y_coordinate = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Frame Utama
main_frame = tk.Frame(root, bg="#222831")
main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

# Header
header_label = tk.Label(main_frame, text="Age & Gender Detector", font=("Arial", 16, "bold"), fg="white", bg="#222831")
header_label.pack(pady=10)

# Tombol-tombol
button_style = {"font": ("Arial", 12), "width": 20, "pady": 5}

browse_button = tk.Button(main_frame, text="Browse Image", command=browse_file, **button_style, bg="#00ADB5", fg="white")
browse_button.pack(pady=5)

webcam_button = tk.Button(main_frame, text="Open Webcam", command=open_webcam, **button_style, bg="#00ADB5", fg="white")
webcam_button.pack(pady=5)

help_button = tk.Button(main_frame, text="Help", command=show_help, **button_style, bg="#FFB400", fg="black")
help_button.pack(pady=5)

quit_button = tk.Button(main_frame, text="Quit", command=quit_app, **button_style, bg="#FF4848", fg="white")
quit_button.pack(pady=5)

# Menjalankan aplikasi
root.mainloop()
