import asyncio
import cv2
from face_recognize import FaceRecognizer
from tts import VoiceAssistant
from qa_model import KnowledgeQA

async def main():
    face = FaceRecognizer()
    voice = VoiceAssistant()
    qa = KnowledgeQA()
    face_detected = False

    while True:
        ret, frame = face.get_frame()
        if not ret:
            break

        recognized ,name= face.recognize_faces(frame)
        for (top, right, bottom, left), name in recognized:
            color = (0, 255, 0) if name != "who?" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if not face_detected:
                
                await voice.speek(f"11{name}我是你的私人专属甘薯助手，你有什么问题吗？")
                face_detected = True

        cv2.imshow('Face Recognition', frame)

        if face_detected:
            print("请你说\n")
            query = voice.listen()
            
            if query:
                answer = qa.ask(query)
                print("模型答案是:\n"+answer)
                await voice.speek("11" + answer)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face.release()

if __name__ == "__main__":
    asyncio.run(main())