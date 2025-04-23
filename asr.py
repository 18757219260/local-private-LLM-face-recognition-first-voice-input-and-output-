import pyaudio
import webrtcvad
import time
from aip import AipSpeech

# Baidu API credentials
APP_ID = '118613302'
API_KEY = '7hSl10mvmtaCndZoab0S3BXQ' 
SECRET_KEY = 'Fv10TxiFLmWb4UTAdLeA2eaTIE56QtkW'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

class ASRhelper:
    def __init__(self):
        # Audio settings
        self.CHUNK = 480  
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.SILENCE_DURATION = 1.0  
        self.MAX_RECORD_SECONDS = 5  
        self.NO_SPEECH_TIMEOUT = 2.0  
        self.voice = "zh-CN-XiaoyiNeural"

        self.vad = webrtcvad.Vad(2)  
     
        self.p, self.stream = self.get_audio_stream()

    def get_audio_stream(self):
        """Initialize and return an audio stream."""
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        return p, stream

    def real_time_recognition(self):
        """Perform real-time speech recognition with VAD."""
        print("*" * 10, "开始实时语音识别，请说话...")

        while True:
            frames = []
            start_time = time.time()
            speech_started = False
            last_speech_time = time.time()

            while True:
                try:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    is_speech = self.vad.is_speech(data, self.RATE)

                    if is_speech:
                        if not speech_started:
                            speech_started = True
                            print("可以说话咯")
                        last_speech_time = time.time()
                        frames.append(data)
                    else:
                        if speech_started:

                            if (time.time() - last_speech_time) >= self.SILENCE_DURATION:
                                print("检测到语音结束")
                                break
                    if (time.time() - start_time) >= self.MAX_RECORD_SECONDS:
                        print("录完了")
                        break

                    if not speech_started and (time.time() - start_time) >= self.NO_SPEECH_TIMEOUT:
                        print("请你提出问题")
                        start_time = time.time()  # Reset start time

                except Exception as e:
                    print("录音有错:", str(e))
                    break

            if frames:
                audio_data = b"".join(frames)
                print(f"Sending {len(audio_data)} bytes of audio data")
                result = client.asr(audio_data, 'pcm', self.RATE, {'dev_pid': 1537})
                if result['err_no'] == 0:
                    print("✅ 用户说：:", result['result'][0])
                else:
                    print("❌ 识别失败:", result['err_msg'], "错误码:", result['err_no'])
            else:
                print("没有录到语音")

    def stop_recording(self):
        """Stop and close the audio stream."""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("音频流已关闭")
    def main(self):
        try:
            self.real_time_recognition()
        except KeyboardInterrupt:
            print("*" * 10, "停止实时语音识别")
        finally:
            assistant.stop_recording()

if __name__ == '__main__':
    assistant = ASRhelper()
    assistant.main()