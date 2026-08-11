# tts.py
# 목적: 인식된 수어 단어(텍스트)를 음성으로 출력
# 방식1: pyttsx3  - 오프라인, 인터넷 불필요, 즉시 재생 (기본 권장)
# 방식2: gTTS     - 구글 TTS, 인터넷 필요, 음질 자연스러움
#
# 설치:
#   pip install pyttsx3          (오프라인용)
#   pip install gTTS playsound==1.2.2   (온라인용, 선택)

# ------------------------------------------------------------
# 방식 1: pyttsx3 (오프라인)
# ------------------------------------------------------------
class OfflineTTS:
    def __init__(self, rate=170, volume=1.0):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)      # 말하기 속도 (기본 200)
        self.engine.setProperty("volume", volume)  # 0.0 ~ 1.0

        # 한국어 음성이 있으면 자동 선택 (윈도우: Heami 등)
        for v in self.engine.getProperty("voices"):
            name = (v.name or "").lower()
            langs = str(getattr(v, "languages", "")).lower()
            if "korean" in name or "ko" in langs or "heami" in name:
                self.engine.setProperty("voice", v.id)
                break

    def speak(self, text):
        """텍스트를 즉시 소리내어 읽음 (읽는 동안 잠깐 멈춤)"""
        if not text:
            return
        self.engine.say(text)
        self.engine.runAndWait()

    def list_voices(self):
        """설치된 음성 목록 확인용"""
        for v in self.engine.getProperty("voices"):
            print(f"- {v.name}  |  id={v.id}")


# ------------------------------------------------------------
# 방식 2: gTTS (온라인, 음질 좋음)
# ------------------------------------------------------------
class OnlineTTS:
    def __init__(self, lang="ko"):
        from gtts import gTTS
        self.gTTS = gTTS
        self.lang = lang

    def speak(self, text):
        if not text:
            return
        import tempfile, os
        from playsound import playsound
        tts = self.gTTS(text=text, lang=self.lang)
        # 임시 mp3로 저장 후 재생
        path = os.path.join(tempfile.gettempdir(), "signbridge_tts.mp3")
        tts.save(path)
        playsound(path)


# ------------------------------------------------------------
# 공통 팩토리: 원하는 방식 하나 만들어 반환
# ------------------------------------------------------------
def make_tts(mode="offline"):
    """mode='offline'(pyttsx3) 또는 'online'(gTTS)"""
    if mode == "online":
        return OnlineTTS()
    return OfflineTTS()


# ------------------------------------------------------------
# 단독 테스트: python tts.py
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("[TTS 테스트]")
    print("=" * 50)

    tts = make_tts("offline")   # 먼저 오프라인으로 테스트

    print(">> 설치된 음성 목록:")
    if isinstance(tts, OfflineTTS):
        tts.list_voices()

    for word in ["안녕하세요", "감사합니다", "걷다", "지도"]:
        print(f">> 재생: {word}")
        tts.speak(word)

    print("[완료] TTS 테스트 종료")
