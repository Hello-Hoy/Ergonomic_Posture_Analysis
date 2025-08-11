from gtts import gTTS
from playsound import playsound
import threading
import os
import time


class FeedbackHandler:
    def __init__(self):
        self.last_feedback_time = 0
        self.feedback_cooldown = 5  # 초 단위, 피드백 간 최소 간격

    def provide_text_feedback(self, message):
        print(f"[피드백]: {message}")

    def _play_audio_task(self, filename):
        try:
            playsound(filename)
            os.remove(filename)  # 재생 후 임시 파일 삭제
        except Exception as e:
            print(f"오디오 재생 오류: {e}")

    def provide_voice_feedback(self, message):
        current_time = time.time()
        if current_time - self.last_feedback_time > self.feedback_cooldown:
            self.last_feedback_time = current_time
            try:
                tts = gTTS(text=message, lang='ko')
                filename = "alert.mp3"
                tts.save(filename)

                # 별도의 스레드에서 오디오 재생 실행
                audio_thread = threading.Thread(target=self._play_audio_task, args=(filename,))
                audio_thread.start()

            except Exception as e:
                print(f"gTTS 오류: {e}")