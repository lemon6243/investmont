# src/ue_bridge.py
# SignBridge → UE 5.8 브리지
#   play  : {type:play, gloss:[...], korean, blend}
#   frame : send_mh_live 와 같은 pts/q  (수신기만 있으면 점이 움직임)
#   caption: 자막

from __future__ import annotations

import json
import os
import socket
import time
from typing import Iterable, Optional

UE_IP = "127.0.0.1"
UE_PORT = 7755
FPS = 30

ARM_BONES = {
    "upperarm_l", "lowerarm_l", "hand_l",
    "upperarm_r", "lowerarm_r", "hand_r",
}


class UeBridge:
    def __init__(self, ip: str = UE_IP, port: int = UE_PORT):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ok = True
        self.last_error = ""

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def send(self, payload: dict) -> bool:
        try:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.sock.sendto(raw, self.addr)
            self.ok = True
            self.last_error = ""
            return True
        except OSError as e:
            self.ok = False
            self.last_error = str(e)
            return False

    def ping(self) -> bool:
        return self.send({"type": "hello", "src": "signbridge"})

    def caption(self, text: str, gloss: Optional[Iterable[str]] = None) -> bool:
        return self.send({
            "type": "caption",
            "korean": text,
            "gloss": list(gloss or []),
        })

    def play(self, gloss: list[str], korean: str = "", blend: float = 0.18) -> bool:
        return self.send({
            "type": "play",
            "gloss": list(gloss),
            "korean": korean,
            "blend": float(blend),
            "fps": FPS,
        })

    def send_frame(self, pts, q: dict, word: str = "", i: int = 0, n: int = 0) -> bool:
        return self.send({
            "type": "frame",
            "word": word,
            "i": int(i),
            "n": int(n),
            "space": "bone",      # 추가
            "mode": "replace",    # 추가
            "pts": [float(x) for x in pts],
            "q": q,
        })


def try_load_3d_clip(word: str | None = None):
    """있으면 (ue_seq, bones, word) 반환. 없으면 (None, None, None)."""
    try:
        from send_mh_live import (
            load_from_npy,
            load_from_json_folder,
            seq_to_ue,
        )
        from pos_to_rot import positions_to_local_quats
    except Exception:
        return None, None, None

    seq, label = load_from_npy(word or None, None)
    if seq is None:
        seq, label = load_from_json_folder()
    if seq is None:
        return None, None, None
    ue = seq_to_ue(seq)
    bones, _root = positions_to_local_quats(ue)
    return ue, bones, label


def stream_clip(bridge: UeBridge, ue, bones, word: str, arms_only: bool = True,
                stop_flag=None, fps: float = FPS):
    """한 클립을 30fps로 한 번 재생. stop_flag() 가 True면 중단."""
    T = int(ue.shape[0])
    dt = 1.0 / max(fps, 1.0)
    for t in range(T):
        if stop_flag is not None and stop_flag():
            return False
        q = {}
        for name, arr in bones.items():
            if arms_only and name not in ARM_BONES:
                continue
            q[name] = [round(float(x), 5) for x in arr[t]]
        bridge.send_frame(ue[t].reshape(-1), q, word=word, i=t, n=T)
        time.sleep(dt)
    return True
