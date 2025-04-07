import cv2
import face_recognition

class FaceRecognizer:
    def __init__(self):
        self.known_faces = {
            name: face_recognition.face_encodings(
                face_recognition.load_image_file(f"images/{file}"
            ))[0] for name, file in {
                "wudawang": "zhenyu.jpg",
                "jiazhuo": "jiazhuo.jpeg",
                "yuhui": "yuhui.jpeg",
                "dongyihao": "yihao.jpeg",
                "cjh": "jianhao.jpeg",
            }.items()
        }
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def recognize_faces(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(list(self.known_faces.values()), face_encoding)
            name = "who?"
            if True in matches:
                first_match_index = matches.index(True)
                name = list(self.known_faces.keys())[first_match_index]
                #print(name+" 你好！")
            results.append(((top, right, bottom, left), name))
        return results,name

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = FaceRecognizer()

    try:
        while True:
            ret, frame = recognizer.get_frame()
            if not ret:
                print("无法获取视频帧")
                break

            results, name = recognizer.recognize_faces(frame)
            for (top, right, bottom, left), name in results:
                color = (0, 255, 0) if name != "who?" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        recognizer.release()