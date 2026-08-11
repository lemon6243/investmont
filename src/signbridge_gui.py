# signbridge_gui.py
import sys
import os
import subprocess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QLineEdit, QStatusBar, QSizePolicy,
    QDialog, QPlainTextEdit, QScrollArea, QGridLayout, QInputDialog, QMessageBox
)
from PySide6.QtCore import Qt, QProcess

# ── 색상 팔레트 ──────────────────────────────────────
BG        = "#0f1117"
CARD      = "#1a1d29"
CARD_LINE = "#2a2e3d"
TEXT      = "#e6e8ef"
SUBTEXT   = "#8b90a3"
ACCENT    = "#5b8cff"
ACCENT2   = "#00d0a3"
DANGER    = "#ff5c72"

# 관리자 PIN (원하면 변경). None 으로 두면 PIN 확인 생략.
ADMIN_PIN = "1234"

# 이 GUI 파일이 있는 폴더(=src) 기준으로 스크립트 실행
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 관리자 패널에 표시할 스크립트 (카테고리별) ──────────
#   (표시이름, 파일명, 설명)
ADMIN_SCRIPTS = {
    "데모 · 추론": [
        ("실시간 수어 인식 + TTS", "webcam_infer.py", "웹캠 → 수어 인식 → 문장 → 음성"),
        ("텍스트 → 수어 아바타",   "text_to_sign.py", "타이핑 → 글로스 → 캐릭터 재생"),
        ("아바타 프리뷰",          "avatar_preview.py", "저장된 키포인트 → 캐릭터 렌더링"),
        ("단어 → 문장 변환 테스트", "sentence_builder.py", "규칙+LLM 문장 조립 테스트"),
        ("텍스트 → 글로스 변환",    "text_to_gloss.py", "한국어 → 글로스 변환 테스트"),
        ("TTS 단독 테스트",        "tts.py", "음성 출력 엔진 단독 실행"),
    ],
    "데이터 수집 · 처리": [
        ("웹캠 데이터 수집",    "collect_webcam.py", "단어별 수어 녹화 저장"),
        ("전체 클립 누적",      "collect_all.py", "AI Hub 클립 → dataset_all 누적"),
        ("키포인트 추출",       "extract_keypoints.py", "영상 → 키포인트 추출"),
        ("NPY 빌드",           "build_npy.py", "키포인트 → 학습용 npy"),
        ("최종셋 빌드",        "build_final.py", "최종 학습 데이터셋 구성"),
        ("데이터셋 로드 확인",  "load_dataset.py", "데이터셋 로딩 검증"),
    ],
    "학습": [
        ("모델 학습",          "train.py", "LSTM 모델 학습 실행"),
    ],
    "점검 · 유틸": [
        ("환경 점검",          "check_setup.py", "패키지/환경 설치 확인"),
        ("데이터 점검",        "check_data.py", "데이터 무결성 확인"),
        ("좌우 반전 점검",      "check_lr.py", "swap_lr / flip 검증"),
        ("실용단어 검색",       "search_words.py", "AI Hub 단어 존재 검색"),
    ],
}


STYLE = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
    font-size: 14px;
}}
QLabel#Title {{ font-size: 26px; font-weight: 800; color: {TEXT}; padding: 4px 2px; }}
QLabel#Subtitle {{ font-size: 13px; color: {SUBTEXT}; }}
QFrame#Card {{ background-color: {CARD}; border: 1px solid {CARD_LINE}; border-radius: 16px; }}
QLabel#PanelHeader {{ font-size: 16px; font-weight: 700; color: {TEXT}; }}
QLabel#PanelBadge {{ font-size: 12px; font-weight: 700; color: white; background-color: {ACCENT}; border-radius: 10px; padding: 3px 10px; }}
QLabel#PanelBadgeB {{ background-color: {ACCENT2}; }}
QLabel#VideoArea {{ background-color: #0b0d14; border: 2px dashed {CARD_LINE}; border-radius: 12px; color: {SUBTEXT}; font-size: 14px; }}
QLabel#ResultBox {{ background-color: #0b0d14; border: 1px solid {CARD_LINE}; border-radius: 10px; color: {TEXT}; font-size: 18px; font-weight: 600; padding: 14px; }}
QLineEdit {{ background-color: #0b0d14; border: 1px solid {CARD_LINE}; border-radius: 10px; padding: 10px 12px; color: {TEXT}; font-size: 15px; }}
QLineEdit:focus {{ border: 1px solid {ACCENT}; }}
QPushButton {{ background-color: {ACCENT}; color: white; border: none; border-radius: 10px; padding: 11px 18px; font-size: 14px; font-weight: 700; }}
QPushButton:hover {{ background-color: #6f9bff; }}
QPushButton:pressed {{ background-color: #4a76e0; }}
QPushButton#Ghost {{ background-color: transparent; color: {TEXT}; border: 1px solid {CARD_LINE}; }}
QPushButton#Ghost:hover {{ border: 1px solid {ACCENT}; color: {ACCENT}; }}
QPushButton#Admin {{ background-color: transparent; color: {SUBTEXT}; border: 1px solid {CARD_LINE}; padding: 6px 14px; font-size: 12px; }}
QPushButton#Admin:hover {{ border: 1px solid {ACCENT}; color: {ACCENT}; }}
QStatusBar {{ background-color: {CARD}; color: {SUBTEXT}; border-top: 1px solid {CARD_LINE}; font-size: 12px; }}
"""

ADMIN_STYLE = f"""
QDialog {{ background-color: {BG}; color: {TEXT}; font-family: 'Segoe UI', 'Malgun Gothic', sans-serif; }}
QLabel#CatHeader {{ font-size: 15px; font-weight: 700; color: {ACCENT2}; padding-top: 8px; }}
QPushButton {{ background-color: {CARD}; color: {TEXT}; border: 1px solid {CARD_LINE};
    border-radius: 10px; padding: 12px; font-size: 13px; font-weight: 600; text-align: left; }}
QPushButton:hover {{ border: 1px solid {ACCENT}; color: {ACCENT}; }}
QPushButton#Stop {{ background-color: transparent; color: {DANGER}; border: 1px solid {DANGER}; font-weight: 700; }}
QPushButton#Stop:hover {{ background-color: {DANGER}; color: white; }}
QPlainTextEdit {{ background-color: #0b0d14; color: #b8ffcf; border: 1px solid {CARD_LINE};
    border-radius: 10px; font-family: 'Consolas', monospace; font-size: 12px; }}
QScrollArea {{ border: none; }}
"""


def make_card():
    card = QFrame()
    card.setObjectName("Card")
    card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return card


# ══════════════════════════════════════════════════════
#  관리자 콘솔 (별도 창)
# ══════════════════════════════════════════════════════
class AdminConsole(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SignBridge · 개발자 콘솔")
        self.resize(920, 640)
        self.setStyleSheet(ADMIN_STYLE)
        self.processes = []  # 실행 중인 QProcess 보관

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        # ── 왼쪽: 스크립트 버튼 목록 (스크롤) ──
        left = QVBoxLayout()
        left.setSpacing(6)
        title = QLabel("개발자 스크립트")
        title.setStyleSheet(f"font-size:18px; font-weight:800; color:{TEXT};")
        left.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        grid = QVBoxLayout(inner)
        grid.setSpacing(6)

        for category, items in ADMIN_SCRIPTS.items():
            cat = QLabel(category)
            cat.setObjectName("CatHeader")
            grid.addWidget(cat)
            for label, filename, desc in items:
                btn = QPushButton(f"  {label}\n  {desc}")
                btn.setToolTip(f"{filename}\n{desc}")
                btn.clicked.connect(lambda _=False, f=filename, l=label: self.run_script(f, l))
                grid.addWidget(btn)
        grid.addStretch()
        scroll.setWidget(inner)
        left.addWidget(scroll, stretch=1)
        root.addLayout(left, stretch=3)

        # ── 오른쪽: 로그 출력 + 컨트롤 ──
        right = QVBoxLayout()
        right.setSpacing(8)
        log_title = QLabel("실행 로그")
        log_title.setStyleSheet(f"font-size:15px; font-weight:700; color:{TEXT};")
        right.addWidget(log_title)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        right.addWidget(self.log, stretch=1)

        ctrl = QHBoxLayout()
        clear_btn = QPushButton("로그 지우기")
        clear_btn.clicked.connect(self.log.clear)
        stop_btn = QPushButton("실행 중 전체 종료")
        stop_btn.setObjectName("Stop")
        stop_btn.clicked.connect(self.stop_all)
        ctrl.addWidget(clear_btn)
        ctrl.addWidget(stop_btn)
        right.addLayout(ctrl)
        root.addLayout(right, stretch=4)

        self._log(f"작업 디렉토리: {BASE_DIR}")
        self._log(f"파이썬 실행기: {sys.executable}")

    def _log(self, text):
        self.log.appendPlainText(text)

    def run_script(self, filename, label):
        path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(path):
            self._log(f"[오류] 파일을 찾을 수 없음: {path}")
            QMessageBox.warning(self, "실행 실패", f"{filename} 파일이 없습니다.")
            return

        self._log(f"\n▶ 실행: {label}  ({filename})")
        proc = QProcess(self)
        proc.setWorkingDirectory(BASE_DIR)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        # 현재 venv 의 python 그대로 사용 (sys.executable)
        proc.readyReadStandardOutput.connect(lambda p=proc: self._read_output(p))
        proc.finished.connect(lambda code, _s, l=label: self._log(f"■ 종료: {l} (code={code})"))
        proc.start(sys.executable, [path])
        self.processes.append(proc)

    def _read_output(self, proc):
        data = proc.readAllStandardOutput().data()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("cp949", errors="replace")  # 윈도우 한글 콘솔 대응
        self.log.moveCursor(self.log.textCursor().End)
        self.log.insertPlainText(text)

    def stop_all(self):
        alive = 0
        for p in self.processes:
            if p.state() != QProcess.NotRunning:
                p.kill()
                alive += 1
        self._log(f"\n[중지] 실행 중이던 프로세스 {alive}개 종료")

    def closeEvent(self, event):
        self.stop_all()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════
#  메인 화면
# ══════════════════════════════════════════════════════
class SignBridgeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignBridge")
        self.resize(1180, 720)
        self.setStyleSheet(STYLE)
        self.admin_window = None

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 20, 24, 12)
        outer.setSpacing(16)

        # ── 헤더 (제목 + 관리자 버튼) ──
        header = QHBoxLayout()
        head_text = QVBoxLayout()
        head_text.setSpacing(2)
        title = QLabel("SignBridge")
        title.setObjectName("Title")
        subtitle = QLabel("실시간 양방향 한국수어 소통 시스템  ·  병원 · 은행 창구 데모")
        subtitle.setObjectName("Subtitle")
        head_text.addWidget(title)
        head_text.addWidget(subtitle)
        header.addLayout(head_text)
        header.addStretch()

        admin_btn = QPushButton("⚙  관리자")
        admin_btn.setObjectName("Admin")
        admin_btn.clicked.connect(self.open_admin)
        header.addWidget(admin_btn, alignment=Qt.AlignTop)
        outer.addLayout(header)

        # ── 좌우 패널 ──
        panels = QHBoxLayout()
        panels.setSpacing(16)
        panels.addWidget(self._build_panel_a())
        panels.addWidget(self._build_panel_b())
        outer.addLayout(panels, stretch=1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("준비 완료  ·  대기 중")

    # 관리자 콘솔 열기 (PIN 확인)
    def open_admin(self):
        if ADMIN_PIN is not None:
            pin, ok = QInputDialog.getText(
                self, "관리자 인증", "PIN 을 입력하세요:", QLineEdit.Password
            )
            if not ok:
                return
            if pin != ADMIN_PIN:
                QMessageBox.warning(self, "인증 실패", "PIN 이 올바르지 않습니다.")
                self.status.showMessage("관리자 인증 실패")
                return

        if self.admin_window is None:
            self.admin_window = AdminConsole(self)
        self.admin_window.show()
        self.admin_window.raise_()
        self.admin_window.activateWindow()
        self.status.showMessage("관리자 콘솔 열림")

    def _build_panel_a(self):
        card = make_card()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(14)
        head = QHBoxLayout()
        h = QLabel("수어 → 텍스트 · 음성")
        h.setObjectName("PanelHeader")
        badge = QLabel("A")
        badge.setObjectName("PanelBadge")
        head.addWidget(h)
        head.addStretch()
        head.addWidget(badge)
        lay.addLayout(head)
        video = QLabel("📷  카메라 영상 영역\n(웹캠 연결 시 표시)")
        video.setObjectName("VideoArea")
        video.setAlignment(Qt.AlignCenter)
        video.setMinimumHeight(280)
        lay.addWidget(video, stretch=1)
        self.result_a = QLabel("인식된 문장이 여기에 표시됩니다.")
        self.result_a.setObjectName("ResultBox")
        self.result_a.setWordWrap(True)
        self.result_a.setMinimumHeight(80)
        lay.addWidget(self.result_a)
        btns = QHBoxLayout()
        btns.setSpacing(10)
        start = QPushButton("▶  인식 시작")
        start.clicked.connect(lambda: self.status.showMessage("[미연결] 수어 인식 기능 연결 예정"))
        speak = QPushButton("🔊  음성 출력")
        speak.setObjectName("Ghost")
        speak.clicked.connect(lambda: self.status.showMessage("[미연결] TTS 연결 예정"))
        btns.addWidget(start)
        btns.addWidget(speak)
        btns.addStretch()
        lay.addLayout(btns)
        return card

    def _build_panel_b(self):
        card = make_card()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(14)
        head = QHBoxLayout()
        h = QLabel("텍스트 → 수어 아바타")
        h.setObjectName("PanelHeader")
        badge = QLabel("B")
        badge.setObjectName("PanelBadgeB")
        head.addWidget(h)
        head.addStretch()
        head.addWidget(badge)
        lay.addLayout(head)
        avatar = QLabel("🧑  아바타 표시 영역\n(텍스트 입력 후 재생)")
        avatar.setObjectName("VideoArea")
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setMinimumHeight(280)
        lay.addWidget(avatar, stretch=1)
        self.input_b = QLineEdit()
        self.input_b.setPlaceholderText("변환할 한국어 문장을 입력하세요 (예: 머리가 아파요)")
        lay.addWidget(self.input_b)
        btns = QHBoxLayout()
        btns.setSpacing(10)
        play = QPushButton("▶  수어로 변환")
        play.clicked.connect(lambda: self.status.showMessage("[미연결] 텍스트→수어 재생 연결 예정"))
        clear = QPushButton("지우기")
        clear.setObjectName("Ghost")
        clear.clicked.connect(lambda: self.input_b.clear())
        btns.addWidget(play)
        btns.addWidget(clear)
        btns.addStretch()
        lay.addLayout(btns)
        return card


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SignBridgeGUI()
    win.show()
    sys.exit(app.exec())
