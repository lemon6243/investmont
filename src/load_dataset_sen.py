# load_dataset_sen.py
# 목적: AI Hub SEN(문장) 데이터를 가변 길이 시퀀스로 로딩
# WORD 로더와 다른 점:
#   - data[0]만 쓰지 않고 모든 형태소 구간을 사용
#   - 라벨은 한국어 문장 + 글로스 리스트
#   - 60프레임으로 자르지 않음 (문장이 뭉개지지 않게)
# 키포인트 변환은 keypoint_schema_3d 재사용

import os
import json
import numpy as np

from keypoint_schema_3d import (
    openpose_to_common_3d,
    fill_low_conf_sequence,
    normalize_3d,
    flatten_3d,
    N_KEYPOINTS,
    FEATURE_DIM,
)
from load_dataset_3d import scan_video_dirs, find_morpheme_files

FPS = 30

# 배포본마다 한국어 문장 키 이름이 다를 수 있어 후보를 넓게 둠
_KOREAN_KEYS = (
    "korean", "kor", "sentence", "text", "korean_text",
    "ko", "utterance", "script", "origin", "original",
    "korean_sentence", "sent",
)


def _attr_name(attr):
    if isinstance(attr, dict):
        return attr.get("name") or attr.get("value") or attr.get("label")
    return str(attr) if attr else None


def parse_sentence_morpheme(m):
    """
    SEN morpheme JSON 1개 -> dict 또는 None
      name    : 영상/폴더 이름 (확장자 제거)
      korean  : 한국어 문장 (키를 못 찾으면 글로스를 공백으로 이은 값)
      gloss   : ["머리", "아프다", ...]
      start   : 첫 형태소 start (초)
      end     : 마지막 형태소 end (초)
      segs    : [{word, start, end}, ...]
      korean_from_meta : 한국어를 meta에서 찾았는지
    """
    meta = m.get("metaData") or m.get("metadata") or {}
    raw_name = meta.get("name") or m.get("name") or ""
    name = str(raw_name).replace(".mp4", "").replace(".MP4", "")
    if not name:
        return None

    segs_raw = m.get("data") or []
    if not segs_raw:
        return None

    segs = []
    for seg in segs_raw:
        attrs = seg.get("attributes") or []
        word = _attr_name(attrs[0]) if attrs else None
        if not word:
            continue
        try:
            s = float(seg.get("start", 0))
            e = float(seg.get("end", 0))
        except (TypeError, ValueError):
            continue
        segs.append({"word": str(word), "start": s, "end": e})

    if not segs:
        return None

    gloss = [s["word"] for s in segs]
    start = min(s["start"] for s in segs)
    end = max(s["end"] for s in segs)

    korean = None
    for src in (meta, m):
        for k in _KOREAN_KEYS:
            v = src.get(k)
            if isinstance(v, str) and v.strip():
                korean = v.strip()
                break
        if korean:
            break

    return {
        "name": name,
        "korean": korean or " ".join(gloss),
        "gloss": gloss,
        "start": start,
        "end": end,
        "segs": segs,
        "korean_from_meta": korean is not None,
    }


def peek_morpheme(morpheme_root, n=1):
    """SEN 받은 직후: 실제 키 이름을 확인."""
    files = find_morpheme_files(morpheme_root)
    print(">> morpheme 파일 수:", len(files))
    for fp in files[:n]:
        with open(fp, "r", encoding="utf-8") as f:
            m = json.load(f)
        print("=" * 60)
        print("file:", fp)
        print("top keys:", list(m.keys()))
        meta = m.get("metaData") or m.get("metadata") or {}
        print("metaData keys:", list(meta.keys()))
        data = m.get("data") or []
        print("data 개수:", len(data))
        if data:
            print("data[0] keys:", list(data[0].keys()))
            print("data[0]:", json.dumps(data[0], ensure_ascii=False)[:400])
        parsed = parse_sentence_morpheme(m)
        print("parsed:", parsed)


def _list_keypoint_files(video_dir):
    files = []
    for dp, dn, fns in os.walk(video_dir):
        for fn in fns:
            if fn.endswith("_keypoints.json"):
                files.append(os.path.join(dp, fn))
    return sorted(files)


def load_sentence_sequence(video_dir, start_sec, end_sec):
    """
    문장 구간을 가변 길이로 로드. 60프레임 리샘플/패딩 없음.
    반환:
      x_raw  (T, 50, 3)
      x_norm (T, 150)
    실패 시 (None, None)
    """
    files = _list_keypoint_files(video_dir)
    if not files:
        return None, None

    s = int(round(start_sec * FPS))
    e = int(round(end_sec * FPS))
    s = max(0, s)
    e = min(len(files), max(e, s + 1))
    if e <= s:
        s, e = 0, len(files)

    kp_list, conf_list = [], []
    for fp in files[s:e]:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        people = d.get("people")
        if people is None:
            continue
        kp, conf = openpose_to_common_3d(people)
        kp_list.append(kp)
        conf_list.append(conf)

    if not kp_list:
        return None, None

    seq_kp = np.stack(kp_list)        # (T, 50, 3)
    seq_conf = np.stack(conf_list)    # (T, 50)
    seq_kp = fill_low_conf_sequence(seq_kp, seq_conf)

    x_raw = seq_kp.astype(np.float32)
    x_norm = np.stack(
        [flatten_3d(normalize_3d(f)) for f in seq_kp]
    ).astype(np.float32)
    return x_raw, x_norm


if __name__ == "__main__":
    # SEN 받은 뒤 경로만 맞추고 실행해서 키를 확인
    MORPHEME = r"C:\Users\LG\Downloads\수어 영상\1.Training\sen\[라벨]01_real_sen_morpheme"
    peek_morpheme(MORPHEME, n=1)
