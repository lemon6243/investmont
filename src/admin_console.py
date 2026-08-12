# admin_console.py
import sys, os, subprocess
from PySide6.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QScrollArea, QPlainTextEdit
)
from PySide6.QtCore import QProcess

BG="#0f1117"; CARD="#1a1d29"; CARD_LINE="#2a2e3d"; TEXT="#e6e8ef"
ACCENT="#5b8cff"; ACCENT2="#00d0a3"; DANGER="#ff5c72"

CONSOLE_SCRIPTS = {
    "text_to_sign.py", "sentence_builder.py", "text_to_gloss.py",
    "collect_webcam.py", "collect_all.py", "extract_keypoints.py",
    "build_npy.py", "build_final.py", "train.py", "check_setup.py",
    "check_data.py", "check_lr.py", "search_words.py", "load_dataset.py",
}

ADMIN_SCRIPTS = {
    "데모 · 추론": [
        ("실시간 수어 인식 + TTS", "webcam_infer.py", "웹캠 → 수어 인식 → 문장 → 음성"),
        ("텍스트 → 수어 아바타", "text_to_sign.py", "타이핑 → 글로스 → 캐릭터 재생"),
        ("아바타 프리뷰", "avatar_preview.py", "저장된 키포인트 → 캐릭터 렌더링"),
        ("단어 → 문장 변환 테스트", "sentence_builder.py", "규칙+LLM 문장 조립"),
        ("텍스트 → 글로스 변환", "text_to_gloss.py", "한국어 → 글로스 변환"),
        ("TTS 단독 테스트", "tts.py", "음성 출력 엔진 단독 실행"),
    ],
    "데이터 수집 · 처리": [
        ("웹캠 데이터 수집", "collect_webcam.py", "단어별 수어 녹화 저장"),
        ("전체 클립 누적", "collect_all.py", "AI Hub 클립 → dataset_all 누적"),
        ("키포인트 추출", "extract_keypoints.py", "영상 → 키포인트 추출"),
        ("NPY 빌드", "build_npy.py", "키포인트 → 학습용 npy"),
        ("최종셋 빌드", "build_final.py", "최종 학습 데이터셋 구성"),
        ("데이터셋 로드 확인", "load_dataset.py", "데이터셋 로딩 검증"),
    ],
    "학습": [("모델 학습", "train.py", "LSTM 모델 학습 실행")],
    "점검 · 유틸": [
        ("환경 점검", "check_setup.py", "패키지/환경 설치 확인"),
        ("데이터 점검", "check_data.py", "데이터 무결성 확인"),
        ("좌우 반전 점검", "check_lr.py", "swap_lr / flip 검증"),
        ("실용단어 검색", "search_words.py", "AI Hub 단어 존재 검색"),
    ],
}

ADMIN_STYLE = f"""
QDialog {{ background-color:{BG}; color:{TEXT}; font-family:'Segoe UI','Malgun Gothic',sans-serif; }}
QLabel#CatHeader {{ font-size:15px; font-weight:700; color:{ACCENT2}; padding-top:8px; }}
QPushButton {{ background-color:{CARD}; color:{TEXT}; border:1px solid {CARD_LINE};
    border-radius:10px; padding:12px; font-size:13px; font-weight:600; text-align:left; }}
QPushButton:hover {{ border:1px solid {ACCENT}; color:{ACCENT}; }}
QPushButton#Stop {{ background-color:transparent; color:{DANGER}; border:1px solid {DANGER}; font-weight:700; }}
QPushButton#Stop:hover {{ background-color:{DANGER}; color:white; }}
QPlainTextEdit {{ background-color:#0b0d14; color:#b8ffcf; border:1px solid {CARD_LINE};
    border-radius:10px; font-family:'Consolas',monospace; font-size:12px; }}
QScrollArea {{ border:none; }}
"""


class AdminConsole(QDialog):
    def __init__(self, parent, base_dir):
        super().__init__(parent)
        self.base_dir = base_dir
        self.setWindowTitle("SignBridge · 개발자 콘솔")
        self.resize(920, 640)
        self.setStyleSheet(ADMIN_STYLE)
        self.processes = []

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16); root.setSpacing(14)

        left = QVBoxLayout(); left.setSpacing(6)
        t = QLabel("개발자 스크립트")
        t.setStyleSheet(f"font-size:18px; font-weight:800; color:{TEXT};")
        left.addWidget(t)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); grid = QVBoxLayout(inner); grid.setSpacing(6)
        for cat, items in ADMIN_SCRIPTS.items():
            c = QLabel(cat); c.setObjectName("CatHeader"); grid.addWidget(c)
            for label, fn, desc in items:
                mark = "  ⧉" if fn in CONSOLE_SCRIPTS else ""
                b = QPushButton(f"  {label}{mark}\n  {desc}")
                b.setToolTip(f"{fn}\n{desc}")
                b.clicked.connect(lambda _=False, f=fn, l=label: self.run_script(f, l))
                grid.addWidget(b)
        grid.addStretch(); scroll.setWidget(inner)
        left.addWidget(scroll, stretch=1)
        root.addLayout(left, stretch=3)

        right = QVBoxLayout(); right.setSpacing(8)
        lt = QLabel("실행 로그  (⧉ = 별도 콘솔 창)")
        lt.setStyleSheet(f"font-size:14px; font-weight:700; color:{TEXT};")
        right.addWidget(lt)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        right.addWidget(self.log, stretch=1)
        ctrl = QHBoxLayout()
        cb = QPushButton("로그 지우기"); cb.clicked.connect(self.log.clear)
        sb = QPushButton("실행 중 전체 종료"); sb.setObjectName("Stop")
        sb.clicked.connect(self.stop_all)
        ctrl.addWidget(cb); ctrl.addWidget(sb)
        right.addLayout(ctrl)
        root.addLayout(right, stretch=4)

        self._log(f"작업 디렉토리: {self.base_dir}")
        self._log(f"파이썬 실행기: {sys.executable}")

    def _log(self, text, raw=False):
        if raw:
            self.log.moveCursor(self.log.textCursor().End)
            self.log.insertPlainText(text)
        else:
            self.log.appendPlainText(text)

    def run_script(self, filename, label):
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            self._log(f"\n[오류] 파일 없음: {filename}")
            return
        if filename in CONSOLE_SCRIPTS:
            if os.name == "nt":
                subprocess.Popen(['cmd', '/c', 'start', '', 'cmd', '/k',
                                  sys.executable, path], cwd=self.base_dir)
            else:
                subprocess.Popen([sys.executable, path], cwd=self.base_dir)
            self._log(f"\n[실행] 별도 콘솔에서 {filename} 시작  ({label})")
        else:
            self._log(f"\n▶ 실행: {label}  ({filename})")
            proc = QProcess(self)
            proc.setWorkingDirectory(self.base_dir)
            proc.setProcessChannelMode(QProcess.MergedChannels)
            proc.readyReadStandardOutput.connect(lambda p=proc: self._read(p))
            proc.finished.connect(lambda c, _s, f=filename: self._log(f"\n■ 종료: {f} (code={c})"))
            proc.start(sys.executable, [path])
            self.processes.append(proc)

    def _read(self, proc):
        data = proc.readAllStandardOutput().data()
        try:
            txt = data.decode("utf-8")
        except UnicodeDecodeError:
            txt = data.decode("cp949", errors="replace")
        self.log.moveCursor(self.log.textCursor().End)
        self.log.insertPlainText(txt)

    def stop_all(self):
        n = 0
        for p in self.processes:
            if p.state() != QProcess.NotRunning:
                p.kill(); n += 1
        self._log(f"\n[중지] GUI 프로세스 {n}개 종료 (별도 콘솔 창은 각 창에서 닫으세요)")

    def closeEvent(self, event):
        self.stop_all()
        super().closeEvent(event)
