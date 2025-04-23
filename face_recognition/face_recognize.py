import cv2
import face_recognition
import numpy as np
import time
import pickle

class FaceRecognizer:
    def __init__(self):
        self.known_faces = {}
        self.start_time = time.time()  # 记录开始时间用于10秒超时
        self._load_known_faces()
        self._init_camera()

    def _load_known_faces(self):
        """加载人脸数据库"""
        try:
            with open("/home/wuye/vscode/chatbox/face_recognition/face_model.pkl", "rb") as f:
                self.known_faces = pickle.load(f)
                print("成功加载人脸数据库！")
        except FileNotFoundError:
            print("未找到人脸数据库，请先运行创建数据库脚本！")
            exit()
    def _init_camera(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(0)  # 0 表示使用默认摄像头
        if not self.cap.isOpened():
            print("无法打开摄像头")
            exit()

    def get_frame(self):
        """从摄像头获取视频帧"""
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def recognize_faces(self, frame):
        """执行人脸识别"""
        face_locations = face_recognition.face_locations(frame, model="cnn", number_of_times_to_upsample=1)
        face_encodings = face_recognition.face_encodings(frame, face_locations, model="large")

        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(
                list(self.known_faces.values()), face_encoding, tolerance=0.4
            )
            name = "who?"
            face_distances = face_recognition.face_distance(
                list(self.known_faces.values()), face_encoding
            )
            if True in matches:
                best_match_index = face_distances.argmin()
                if face_distances[best_match_index] < 0.5:
                    name = list(self.known_faces.keys())[best_match_index]
            results.append(((top, right, bottom, left), name))
        return results

    def release(self):
        """释放资源"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = FaceRecognizer()
    try:
        while True:
            # 检查是否超过10秒
            # if time.time() - recognizer.start_time > 10:
            #     print("运行10秒后自动关闭")
            #     break
            ret, frame = recognizer.get_frame()
            if not ret:
                print("无法获取视频帧")
                break
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = recognizer.recognize_faces(frame)
            for (top, right, bottom, left), name in results:
                color = (0, 255, 0) if name != "who?" else (0, 0, 255)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(display_frame, name, (left, top-30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            cv2.imshow('Face Recognition', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        recognizer.release()
