"""
피드백 유틸리티: 텍스트 출력과 gTTS+pygame을 이용한 음성 안내 제공.
- 콘솔 텍스트 피드백 출력
- gTTS로 생성한 mp3를 임시 파일로 저장 후 별도 스레드에서 재생
- 과도한 반복 방지를 위한 쿨다운 적용
"""
from gtts import gTTS
import threading
import os
import time
import pygame
import tempfile


class FeedbackHandler:
    """
    피드백 처리 클래스.
    - provide_text_feedback: 텍스트 메시지를 콘솔에 출력
    - provide_voice_feedback: 한국어 TTS 음성을 생성·재생(쿨다운 포함)
    내부적으로 pygame.mixer를 초기화하고, 재생은 백그라운드 스레드에서 수행합니다.
    """
    def __init__(self):
        self.last_feedback_time = 0
        self.feedback_cooldown = 10  # 초 단위, 피드백 간 최소 간격
        # pygame mixer 초기화
        pygame.mixer.init()

    def provide_text_feedback(self, message):
        print(f"[피드백]: {message}")

    def _play_audio_task(self, filename):
        try:
            # pygame으로 오디오 재생
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            # 재생 완료까지 대기
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            os.remove(filename)  # 재생 후 임시 파일 삭제
        except Exception as e:
            print(f"오디오 재생 오류: {e}")

    def provide_voice_feedback(self, message):
        current_time = time.time()
        if current_time - self.last_feedback_time > self.feedback_cooldown:
            self.last_feedback_time = current_time
            try:
                tts = gTTS(text=message, lang='ko')

                # 임시 디렉토리에 고유한 파일명으로 저장
                temp_dir = tempfile.gettempdir()
                filename = os.path.join(temp_dir, f"alert_{int(time.time())}.mp3")
                tts.save(filename)

                # 별도의 스레드에서 오디오 재생 실행
                audio_thread = threading.Thread(target=self._play_audio_task, args=(filename,))
                audio_thread.daemon = True  # 메인 프로그램 종료 시 같이 종료
                audio_thread.start()

            except Exception as e:
                print(f"gTTS 오류: {e}")