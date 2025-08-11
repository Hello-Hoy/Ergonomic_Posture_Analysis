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
        self.last_feedback_time = 0  # 마지막으로 음성 피드백을 제공한 시간
        self.feedback_cooldown = 10  # 초 단위, 피드백 간 최소 간격 (10초)
        pygame.mixer.init()  # pygame의 mixer 모듈을 초기화합니다.

    def provide_text_feedback(self, message):
        """콘솔에 텍스트 피드백을 출력합니다."""
        print(f"[피드백]: {message}")

    def _play_audio_task(self, filename):
        """
        별도의 스레드에서 오디오 파일을 재생하는 내부 함수입니다.
        재생이 끝나면 임시 파일을 삭제합니다.
        """
        try:
            pygame.mixer.music.load(filename)  # 오디오 파일을 로드합니다.
            pygame.mixer.music.play()  # 오디오를 재생합니다.

            # 재생이 완료될 때까지 대기합니다.
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            os.remove(filename)  # 재생 후 임시 파일을 삭제합니다.
        except Exception as e:
            print(f"오디오 재생 오류: {e}")

    def provide_voice_feedback(self, message):
        """
        gTTS를 이용해 음성 피드백을 생성하고 재생합니다.
        쿨다운 시간이 지나야만 새로운 피드백을 제공합니다.
        """
        current_time = time.time()
        if current_time - self.last_feedback_time > self.feedback_cooldown:
            self.last_feedback_time = current_time  # 마지막 피드백 시간을 갱신합니다.
            try:
                tts = gTTS(text=message, lang='ko')  # 한국어 음성을 생성합니다.

                # 임시 디렉토리에 고유한 파일명으로 mp3 파일을 저장합니다.
                temp_dir = tempfile.gettempdir()
                filename = os.path.join(temp_dir, f"alert_{int(time.time())}.mp3")
                tts.save(filename)

                # 별도의 스레드에서 오디오 재생을 실행하여 메인 루프가 멈추지 않도록 합니다.
                audio_thread = threading.Thread(target=self._play_audio_task, args=(filename,))
                audio_thread.daemon = True  # 메인 프로그램 종료 시 스레드도 함께 종료되도록 설정
                audio_thread.start()

            except Exception as e:
                print(f"gTTS 오류: {e}")