import cv2
import face_recognition
import os
from PIL import Image
import numpy as np
import time
class FaceRecognizer:

    def __init__(self):
        self.known_faces = {}
        self.cap = None
        self._init_known_faces()  
        self._init_camera()       


    def _init_known_faces(self):
        """优化人脸数据加载方式"""
        face_map = {
            
            "jiazhuo": "images/jiazhuo.jpg",
            "yuhui": "images/yuhui.jpg",
            "dongyihao": "images/yihao.jpg",
            "cjh": "images/jianhao.jpg",
            "wudawang": "images/zhenyu.jpg",
        }
        
        for name, file in face_map.items():
            img_path = os.path.join(file)
            img= Image.open(img_path)
            if img.size[0] > 400 or img.size[1] > 400:
                img.thumbnail((400, 400))
            img = np.array(img)
                
            try:
                
                face_loc = face_recognition.face_locations(img, model="hog")
                encoding = face_recognition.face_encodings(
                    img, 
                    known_face_locations=face_loc,
                    num_jitters=3,   
                    model="large"   
                )[0]
                self.known_faces[name] = encoding
            except Exception as e:
                print(f"🔥 加载 {file} 失败: {str(e)}")

    def _init_camera(self):
        """优化摄像头初始化"""
        self.cap = cv2.VideoCapture(0)
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率

    def get_frame(self):
        
        ret, frame = self.cap.read()
        if ret : 
            return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, None

    def recognize_faces(self, frame):
        """优化识别流程"""
        start=time.time()
        face_locations = face_recognition.face_locations(
            frame, 
            model="hog",         
            number_of_times_to_upsample=1  
        )
        
        face_encodings = face_recognition.face_encodings(
            frame, 
            face_locations,model="large"         
        )

        results = []
        current_names = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            
            matches = face_recognition.compare_faces(
                list(self.known_faces.values()), 
                face_encoding,
                tolerance=0.35
            )
            
            name = "who?"
            face_distances = face_recognition.face_distance(
                list(self.known_faces.values()),
                face_encoding
            )
            
            if True in matches:
                best_match_index = face_distances.argmin()
                if face_distances[best_match_index] < 0.4:  
                    name = list(self.known_faces.keys())[best_match_index]
            top, right, bottom, left = self._convert_coordinates(frame, (top, right, bottom, left))
            
            results.append(((top, right, bottom, left), name))
            current_names.append(name)
            end=time.time()
            print(f"人脸识别识别耗时: {end-start:.2f}秒")
            
        return results, (current_names[0] if current_names else "")

    def _convert_coordinates(self, frame, location):
        """坐标转换适配不同分辨率"""
        height, width = frame.shape[:2]
        scale_x = 640 / width
        scale_y = 480 / height
        return (
            int(location[0] * scale_y),
            int(location[1] * scale_x),
            int(location[2] * scale_y),
            int(location[3] * scale_x)
        )

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

#测试

if __name__ == "__main__":
    recognizer = FaceRecognizer()
    
    try:
        while True:
            ret, frame = recognizer.get_frame()
            if not ret:
                print("无法获取视频帧")
                break
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            results, _ = recognizer.recognize_faces(frame)
            
            for (top, right, bottom, left), name in results:
                color = (0, 255, 0) if name != "who?" else (0, 0, 255)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(display_frame, name, (left, top-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3) 
                
            cv2.imshow('Face Recognition', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        recognizer.release()