import asyncio
import cv2
from face_recognize import FaceRecognizer
from tts import VoiceAssistant
from qa_model_easy import KnowledgeQA

async def main():
  
    face = FaceRecognizer()
    voice = VoiceAssistant()
    qa = KnowledgeQA()
    state = "detecting"

    while True:
        if state == "detecting":
    
            ret, frame = face.get_frame()
            if not ret:
                print("无法获取摄像头帧")
                break

    
            recognized, _ = face.recognize_faces(frame)
            for (top, right, bottom, left), name in recognized:
              
                color = (255, 0, 0) if name != "who?" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

               
                if name != "who?":
                    await voice.speak(f"{name}，我是你的私人专属甘薯助手，你有什么问题吗？")
                    state = "qa" 
               
                    cv2.destroyAllWindows()
                    break  
            if state == "detecting":
                cv2.imshow('Face Recognition', frame)

        elif state == "qa":
     
            while True:
                print("请你说（按'q'结束问答）")
                query = voice.listen()
                if query:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("检测到'q'键，结束问答环节")
                        state = "exit"
                        break
               
                    answer = qa.ask(query)
                    print("模型答案是:\n" + answer)
                    await voice.speak('11'+answer)
                else:
                    answer = "未检测到语音输入，请重试。"
                    print(answer)
                    await voice.speak('11'+answer)

           
        if state == "exit":
            break
    face.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())