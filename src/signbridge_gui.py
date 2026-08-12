# signbridge_gui.py
import sys
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import mediapipe as mp

from keypoint_schema import mediapipe_to_common, normalize, flatten, FEATURE_DIM
from tts import make_tts
from sentence_builder import build_sentence, APILLM

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QLineEdit, QStatusBar, QSizePolicy,
    QDialog, QPlainTextEdit, QScrollArea, QInputDialog, QMessageBox, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

# ── 프로젝트 모듈 재사용 ──
from avatar_preview import draw_frame, CANVAS, FPS
from text_to_gloss import text_to_gloss, GlossLLM
from text_to_sign import build_word_bank, make_neutral_pose, build_full_sequence

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "dataset", "sign_lstm.pt")
SWAP_LR = False
USE_FLIP = False
CONF_THRESHOLD = 0.30
DATA_DIR = os.path.join(BASE_DIR, "dataset_all")
X_PATH = os.path.join(DATA_DIR, "X_all.npy")
Y_PATH = os.path.join(DATA_DIR, "y_all.npy")

BG="#0f1117"; CARD="#1a1d29"; CARD_LINE="#2a2e3d"; TEXT="#e6e8ef"
SUBTEXT="#8b90a3"; ACCENT="#5b8cff"; ACCENT2="#00d0a3"; DANGER="#ff5c72"
ADMIN_PIN = "1234"

STYLE = f"""
QMainWindow, QWidget {{ background-color:{BG}; color:{TEXT};
    font-family:'Segoe UI','Malgun Gothic',sans-serif; font-size:14px; }}
QLabel#Title {{ font-size:26px; font-weight:800; color:{TEXT}; padding:4px 2px; }}
QLabel#Subtitle {{ font-size:13px; color:{SUBTEXT}; }}
QFrame#Card {{ background-color:{CARD}; border:1px solid {CARD_LINE}; border-radius:16px; }}
QLabel#PanelHeader {{ font-size:16px; font-weight:700; color:{TEXT}; }}
QLabel#PanelBadge {{ font-size:12px; font-weight:700; color:white; background-color:{ACCENT}; border-radius:10px; padding:3px 10px; }}
QLabel#PanelBadgeB {{ background-color:{ACCENT2}; }}
QLabel#VideoArea {{ background-color:#0b0d14; border:2px solid {CARD_LINE}; border-radius:12px; color:{SUBTEXT}; font-size:14px; }}
QLabel#ResultBox {{ background-color:#0b0d14; border:1px solid {CARD_LINE}; border-radius:10px; color:{TEXT}; font-size:18px; font-weight:600; padding:14px; }}
QLineEdit {{ background-color:#0b0d14; border:1px solid {CARD_LINE}; border-radius:10px; padding:10px 12px; color:{TEXT}; font-size:15px; }}
QLineEdit:focus {{ border:1px solid {ACCENT}; }}
QPushButton {{ background-color:{ACCENT}; color:white; border:none; border-radius:10px; padding:11px 18px; font-size:14px; font-weight:700; }}
QPushButton:hover {{ background-color:#6f9bff; }}
QPushButton:pressed {{ background-color:#4a76e0; }}
QPushButton#Ghost {{ background-color:transparent; color:{TEXT}; border:1px solid {CARD_LINE}; }}
QPushButton#Ghost:hover {{ border:1px solid {ACCENT}; color:{ACCENT}; }}
QPushButton#Admin {{ background-color:transparent; color:{SUBTEXT}; border:1px solid {CARD_LINE}; padding:6px 14px; font-size:12px; }}
QPushButton#Admin:hover {{ border:1px solid {ACCENT}; color:{ACCENT}; }}
QStatusBar {{ background-color:{CARD}; color:{SUBTEXT}; border-top:1px solid {CARD_LINE}; font-size:12px; }}
"""


def make_card():
    card = QFrame(); card.setObjectName("Card")
    card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return card


def np_to_qpix(img_bgr, target_w=None):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg.copy())
    if target_w:
        pix = pix.scaledToWidth(target_w, Qt.SmoothTransformation)
    return pix


class SignLSTM(nn.Module):
    def __init__(self, in_dim, hidden, n_cls, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_cls))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.fc(h_n[-1])


class RecognitionThread(QThread):
    frame_ready = Signal(np.ndarray)
    word_ready = Signal(str, float)
    sentence_ready = Signal(str)

    def __init__(self, cam_index=0):
        super().__init__()
        self.cam_index = cam_index
        self._running = False
        self._capture_request = False
        self.auto_mode = False
        self.MOTION_ON = 0.015
        self.MOTION_OFF = 0.008
        self.STILL_NEED = 6
        self.MIN_LEN = 15
        self.model = None
        self.idx2label = None
        self.seq_len = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        if not os.path.exists(CKPT_PATH):
            return False
        ckpt = torch.load(CKPT_PATH, map_location=self.device)
        self.seq_len = ckpt["seq_len"]
        self.idx2label = {int(k): v for k, v in ckpt["idx2label"].items()}
        self.model = SignLSTM(ckpt["in_dim"], ckpt["hidden"], ckpt["n_classes"],
                              num_layers=ckpt["num_layers"], dropout=ckpt["dropout"]
                              ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        return True

    def _predict(self, frames):
        if not frames:
            return None, 0.0
        arr = np.stack(frames)
        T = arr.shape[0]
        if T < self.seq_len:
            pad = np.zeros((self.seq_len - T, FEATURE_DIM), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0); length = T
        else:
            arr = arr[-self.seq_len:]; length = self.seq_len
        x = torch.tensor(arr[None, ...], dtype=torch.float32, device=self.device)
        lengths = torch.tensor([length], dtype=torch.long)
        with torch.no_grad():
            prob = torch.softmax(self.model(x, lengths), dim=1)[0]
            conf, idx = prob.max(0)
        return self.idx2label[int(idx)], float(conf)

    def request_capture(self):
        self._capture_request = True

    def set_auto_mode(self, on):
        self.auto_mode = on

    def run(self):
        if not self._load_model():
            self.sentence_ready.emit("[오류] 모델 파일(sign_lstm.pt)이 없습니다")
            return
        mp_holistic = mp.solutions.holistic
        cap = cv2.VideoCapture(self.cam_index)
        self._running = True
        recording = False
        record_frames = []
        prev_vec = None
        is_moving = False
        still_count = 0

        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5,
                                  model_complexity=1) as holistic:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break
                if USE_FLIP:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)

                common = mediapipe_to_common(results, w, h, swap_lr=SWAP_LR)
                vec = flatten(normalize(common))

                rec_text = None

                if self.auto_mode:
                    if prev_vec is not None:
                        motion = float(np.mean(np.abs(vec - prev_vec)))
                        if not is_moving and motion > self.MOTION_ON:
                            is_moving = True
                            record_frames = []
                            still_count = 0
                        if is_moving:
                            record_frames.append(vec)
                            rec_text = f"AUTO {len(record_frames)}"
                            if motion < self.MOTION_OFF:
                                still_count += 1
                            else:
                                still_count = 0
                            if still_count >= self.STILL_NEED:
                                if len(record_frames) >= self.MIN_LEN:
                                    label, conf = self._predict(record_frames)
                                    print(f"[자동예측] {label} conf={conf:.3f}")
                                    if conf >= CONF_THRESHOLD:
                                        self.word_ready.emit(label, conf)
                                is_moving = False
                                still_count = 0
                                record_frames = []
                    prev_vec = vec
                else:
                    if self._capture_request and not recording:
                        recording = True
                        record_frames = []
                        self._capture_request = False
                    if recording:
                        record_frames.append(vec)
                        rec_text = f"REC {len(record_frames)}/{self.seq_len}"
                        if len(record_frames) >= self.seq_len:
                            label, conf = self._predict(record_frames)
                            recording = False
                            record_frames = []
                            print(f"[예측] {label} conf={conf:.3f}")
                            if conf >= CONF_THRESHOLD:
                                self.word_ready.emit(label, conf)

                disp = cv2.flip(frame, 1)
                if rec_text:
                    cv2.putText(disp, rec_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                self.frame_ready.emit(disp)
                self.msleep(15)
        cap.release()

    def stop(self):
        self._running = False
        self.wait(1500)


class SignBridgeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignBridge")
        self.resize(1180, 760)
        self.setStyleSheet(STYLE)

        self.reco = None
        self.sentence_words = []
        try:
            self.tts = make_tts("offline")
        except Exception:
            self.tts = None
        self.sentence_llm = APILLM(model="HCX-DASH-002")

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._next_avatar_frame)
        self.anim_frames = []
        self.anim_idx = 0

        self.bank = None
        self.neutral = None
        self.available_words = set()
        self.gloss_llm = None
        self._load_avatar_data()

        root = QWidget(); self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 20, 24, 12); outer.setSpacing(16)

        header = QHBoxLayout()
        ht = QVBoxLayout(); ht.setSpacing(2)
        title = QLabel("SignBridge"); title.setObjectName("Title")
        sub = QLabel("실시간 양방향 한국수어 소통 시스템  ·  병원 · 은행 창구 데모")
        sub.setObjectName("Subtitle")
        ht.addWidget(title); ht.addWidget(sub)
        header.addLayout(ht); header.addStretch()
        admin_btn = QPushButton("⚙  관리자"); admin_btn.setObjectName("Admin")
        admin_btn.clicked.connect(self.open_admin)
        header.addWidget(admin_btn, alignment=Qt.AlignTop)
        outer.addLayout(header)

        panels = QHBoxLayout(); panels.setSpacing(16)
        panels.addWidget(self._build_panel_a())
        panels.addWidget(self._build_panel_b())
        outer.addLayout(panels, stretch=1)

        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.status.showMessage("준비 완료  ·  대기 중")

    def _load_avatar_data(self):
        try:
            if os.path.exists(X_PATH):
                X = np.load(X_PATH)
                y = np.load(Y_PATH, allow_pickle=True)
                self.bank = build_word_bank(X, y)
                self.neutral = make_neutral_pose(self.bank)
                self.available_words = set(self.bank.keys())
                self.gloss_llm = GlossLLM()
        except Exception as ex:
            print(f"[아바타 데이터 로드 실패] {ex}")

    # ══════ A방향 ══════
    def toggle_camera(self):
        if self.reco is None:
            self.reco = RecognitionThread(0)
            self.reco.frame_ready.connect(self._show_camera_frame)
            self.reco.word_ready.connect(self._on_word)
            self.reco.sentence_ready.connect(self._on_sentence)
            self.reco.set_auto_mode(self.chk_auto.isChecked())
            self.reco.start()
            self.btn_cam.setText("■  인식 정지")
            self.btn_capture.setEnabled(not self.chk_auto.isChecked())
            self.btn_done.setEnabled(True)
            self.sentence_words = []
            self.status.showMessage("인식 실행 중")
        else:
            self.reco.stop()
            self.reco = None
            self.btn_cam.setText("▶  인식 시작")
            self.btn_capture.setEnabled(False)
            self.btn_done.setEnabled(False)
            self.video_a.setText("📷  카메라 대기 중")
            self.status.showMessage("인식 정지")

    def _show_camera_frame(self, frame):
        w = self.video_a.width() - 8
        self.video_a.setPixmap(np_to_qpix(frame, target_w=max(320, w)))

    def _capture_word(self):
        if self.reco is not None:
            self.reco.request_capture()
            self.status.showMessage("수어 동작을 취해주세요  ·  녹화 중...")

    def _toggle_auto(self, state):
        on = bool(state)
        if self.reco is not None:
            self.reco.set_auto_mode(on)
        self.btn_capture.setEnabled(not on and self.reco is not None)
        self.status.showMessage("연속 자동 모드" if on else "단어 캡처(수동) 모드")

    def _on_word(self, label, conf):
        self.sentence_words.append(label)
        self.result_a.setText("단어:  " + " + ".join(self.sentence_words)
                              + f"   (최근 {label} {conf*100:.0f}%)")

    def _finish_sentence(self):
        if not self.sentence_words:
            self.status.showMessage("먼저 단어를 캡처하세요")
            return
        sentence = build_sentence(self.sentence_words, llm=self.sentence_llm)
        self.result_a.setText(sentence)
        if self.tts is not None:
            self.tts.speak(sentence)
        self.status.showMessage(f"문장 완성  ·  {sentence}")
        self.sentence_words = []

    def _on_sentence(self, msg):
        self.result_a.setText(msg)
        self.status.showMessage(msg)

    # ══════ B방향 ══════
    def play_text_to_sign(self):
        text = self.input_b.text().strip()
        if not text:
            self.status.showMessage("문장을 입력하세요")
            return
        if self.bank is None:
            QMessageBox.warning(self, "데이터 없음",
                "dataset_all/X_all.npy 가 없습니다. 먼저 데이터를 만들어주세요.")
            return
        gloss = text_to_gloss(text, self.available_words, llm=self.gloss_llm)
        if not gloss:
            self.status.showMessage("보유 단어로 표현 가능한 글로스가 없습니다")
            self.result_b.setText("(표현 가능한 단어 없음)")
            return
        self.result_b.setText("글로스:  " + "  →  ".join(gloss))
        self.anim_frames = build_full_sequence(gloss, self.bank, self.neutral)
        self.anim_idx = 0
        self.anim_timer.start(int(1000 / FPS))
        self.status.showMessage(f"아바타 재생 중  ·  {len(self.anim_frames)} 프레임")

    def _next_avatar_frame(self):
        if self.anim_idx >= len(self.anim_frames):
            self.anim_timer.stop()
            self.status.showMessage("아바타 재생 완료")
            return
        img = draw_frame(self.anim_frames[self.anim_idx])
        w = self.avatar_b.width() - 8
        self.avatar_b.setPixmap(np_to_qpix(img, target_w=max(320, w)))
        self.anim_idx += 1

    def open_admin(self):
        if ADMIN_PIN is not None:
            pin, ok = QInputDialog.getText(self, "관리자 인증", "PIN 을 입력하세요:", QLineEdit.Password)
            if not ok:
                return
            if pin != ADMIN_PIN:
                QMessageBox.warning(self, "인증 실패", "PIN 이 올바르지 않습니다.")
                return
        from admin_console import AdminConsole
        if not hasattr(self, "admin_window") or self.admin_window is None:
            self.admin_window = AdminConsole(self, BASE_DIR)
        self.admin_window.show()
        self.admin_window.raise_(); self.admin_window.activateWindow()

    def _build_panel_a(self):
        card = make_card(); lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 18, 20, 18); lay.setSpacing(14)
        head = QHBoxLayout()
        h = QLabel("수어 → 텍스트 · 음성"); h.setObjectName("PanelHeader")
        badge = QLabel("A"); badge.setObjectName("PanelBadge")
        head.addWidget(h); head.addStretch(); head.addWidget(badge)
        lay.addLayout(head)
        self.video_a = QLabel("📷  카메라 대기 중")
        self.video_a.setObjectName("VideoArea")
        self.video_a.setAlignment(Qt.AlignCenter)
        self.video_a.setMinimumHeight(300)
        lay.addWidget(self.video_a, stretch=1)
        self.result_a = QLabel("인식된 문장이 여기에 표시됩니다.")
        self.result_a.setObjectName("ResultBox"); self.result_a.setWordWrap(True)
        self.result_a.setMinimumHeight(70)
        lay.addWidget(self.result_a)

        btns = QHBoxLayout(); btns.setSpacing(10)
        self.btn_cam = QPushButton("▶  인식 시작")
        self.btn_cam.clicked.connect(self.toggle_camera)
        self.btn_capture = QPushButton("📸  단어 캡처")
        self.btn_capture.setObjectName("Ghost")
        self.btn_capture.clicked.connect(self._capture_word)
        self.btn_capture.setEnabled(False)
        self.btn_done = QPushButton("✓  문장 완성")
        self.btn_done.setObjectName("Ghost")
        self.btn_done.clicked.connect(self._finish_sentence)
        self.btn_done.setEnabled(False)
        self.chk_auto = QCheckBox("연속 자동")
        self.chk_auto.setStyleSheet(f"color:{TEXT}; font-size:13px;")
        self.chk_auto.stateChanged.connect(self._toggle_auto)
        btns.addWidget(self.btn_cam)
        btns.addWidget(self.btn_capture)
        btns.addWidget(self.btn_done)
        btns.addWidget(self.chk_auto)
        btns.addStretch()
        lay.addLayout(btns)
        return card

    def _build_panel_b(self):
        card = make_card(); lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 18, 20, 18); lay.setSpacing(14)
        head = QHBoxLayout()
        h = QLabel("텍스트 → 수어 아바타"); h.setObjectName("PanelHeader")
        badge = QLabel("B"); badge.setObjectName("PanelBadgeB")
        head.addWidget(h); head.addStretch(); head.addWidget(badge)
        lay.addLayout(head)
        self.avatar_b = QLabel("🧑  아바타 대기 중")
        self.avatar_b.setObjectName("VideoArea")
        self.avatar_b.setAlignment(Qt.AlignCenter)
        self.avatar_b.setMinimumHeight(300)
        lay.addWidget(self.avatar_b, stretch=1)
        self.result_b = QLabel("글로스가 여기에 표시됩니다.")
        self.result_b.setObjectName("ResultBox"); self.result_b.setWordWrap(True)
        self.result_b.setMinimumHeight(50)
        lay.addWidget(self.result_b)
        self.input_b = QLineEdit()
        self.input_b.setPlaceholderText("변환할 한국어 문장을 입력하세요 (예: 머리가 아파요)")
        self.input_b.returnPressed.connect(self.play_text_to_sign)
        lay.addWidget(self.input_b)
        btns = QHBoxLayout(); btns.setSpacing(10)
        play = QPushButton("▶  수어로 변환")
        play.clicked.connect(self.play_text_to_sign)
        clear = QPushButton("지우기"); clear.setObjectName("Ghost")
        clear.clicked.connect(lambda: self.input_b.clear())
        btns.addWidget(play); btns.addWidget(clear); btns.addStretch()
        lay.addLayout(btns)
        return card

    def closeEvent(self, event):
        if self.reco is not None:
            self.reco.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SignBridgeGUI()
    win.show()
    sys.exit(app.exec())
