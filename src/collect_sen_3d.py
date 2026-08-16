# collect_sen_3d.py
# 목적: AI Hub SEN(문장) 클립을 단어 셋과 분리해 누적 저장
# 사용법: SEN 압축을 풀 때마다 실행 -> 추가된 문장 확인 -> 원본 삭제 -> 반복
#
# WORD 수집과 다른 점
#   - dataset_all_3d 를 절대 건드리지 않음
#   - 거대한 npy 하나로 안 쌓음 (문장마다 T가 다름)
#   - 첫 형태소만 자르지 않고 문장 전체 + 글로스 전체를 저장
#
# 결과:
#   dataset_sen_3d/
#     clips/{name}.npy        (T, 50, 3)   아바타용 원본 3D
#     clips_norm/{name}.npy   (T, 150)     나중 인식용(선택)
#     catalog.json            문장 메타
#     processed.json          이미 처리한 클립 이름
#     stats.json              요약

import os
import json
import shutil
import argparse
from collections import Counter

import numpy as np

from load_dataset_3d import scan_video_dirs, find_morpheme_files
from load_dataset_sen import (
    parse_sentence_morpheme,
    load_sentence_sequence,
    peek_morpheme,
    N_KEYPOINTS,
    FEATURE_DIM,
)

# ------------------------------------------------------------
# 설정 — WORD 경로/폴더와 절대 같으면 안 됨
# zip 받은 뒤 실제 폴더 이름에 맞게 수정
# ------------------------------------------------------------
MORPHEME = r"C:\Users\LG\Downloads\수어 영상\1.Training\sen\[라벨]01_real_sen_morpheme"
KEYPOINT = r"C:\Users\LG\Downloads\수어 영상\1.Training\sen\[라벨]01_real_sen_keypoint"

ANGLES = None                 # None=전 각도, ['F']=정면만
OUT_DIR = "dataset_sen_3d"    # dataset_all_3d 와 분리
SAVE_NORM = True              # 인식용 정규화본도 같이 저장
FORBIDDEN_OUT = {"dataset_all_3d", "dataset_all", "dataset"}


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_npy_atomic(path, arr):
    tmp = path + ".tmp.npy"
    try:
        np.save(tmp, arr)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def check_free_space(target_dir, need_bytes, margin=1.5):
    free = shutil.disk_usage(target_dir).free
    required = int(need_bytes * margin)
    print(f">> 디스크 여유: {free/1e9:.2f} GB, 필요(여유 포함): {required/1e9:.2f} GB")
    if free < required:
        raise OSError(
            f"디스크 공간 부족: {free/1e9:.2f} GB 남음, "
            f"약 {required/1e9:.2f} GB 필요."
        )


def warn_if_word_tree(video_map):
    names = list(video_map.keys())[:50]
    n_word = sum(1 for n in names if "_WORD" in n or "WORD" in n)
    n_sen = sum(1 for n in names if "_SEN" in n or "SEN" in n)
    if names and n_word > 0 and n_sen == 0:
        raise SystemExit(
            ">> 중단: keypoint 폴더가 WORD 트리로 보입니다.\n"
            "   SEN zip 을 word 와 다른 폴더에 풀고 KEYPOINT/MORPHEME 경로를 수정하세요.\n"
            f"   예시 폴더: {names[0]}"
        )
    if n_word and n_sen:
        print(">> 경고: WORD 와 SEN 폴더가 같이 스캔됐습니다. 경로는 분리하는 게 안전합니다.")


def scan_sentence_meta(morpheme_files, video_map):
    meta = []
    skipped = 0
    no_korean = 0
    for i, mf in enumerate(morpheme_files):
        if i % 20000 == 0 and i:
            print(f"   ...morpheme {i}/{len(morpheme_files)}")
        with open(mf, "r", encoding="utf-8") as f:
            m = json.load(f)
        parsed = parse_sentence_morpheme(m)
        if parsed is None:
            skipped += 1
            continue
        angle = parsed["name"].split("_")[-1]
        if ANGLES and angle not in ANGLES:
            continue
        if parsed["name"] not in video_map:
            continue
        if not parsed["korean_from_meta"]:
            no_korean += 1
        parsed["angle"] = angle
        parsed["group"] = parsed["name"].rsplit("_", 1)[0]
        parsed["morpheme"] = mf
        meta.append(parsed)
    return meta, skipped, no_korean


def main():
    if os.path.basename(os.path.normpath(OUT_DIR)) in FORBIDDEN_OUT:
        raise SystemExit(f">> 중단: OUT_DIR={OUT_DIR} 는 단어 셋입니다. dataset_sen_3d 를 쓰세요.")

    os.makedirs(OUT_DIR, exist_ok=True)
    clip_dir = os.path.join(OUT_DIR, "clips")
    norm_dir = os.path.join(OUT_DIR, "clips_norm")
    os.makedirs(clip_dir, exist_ok=True)
    if SAVE_NORM:
        os.makedirs(norm_dir, exist_ok=True)

    catalog_path = os.path.join(OUT_DIR, "catalog.json")
    processed_path = os.path.join(OUT_DIR, "processed.json")
    stats_path = os.path.join(OUT_DIR, "stats.json")

    print(">> keypoint 클립 폴더 스캔 중...")
    print(">> MORPHEME:", MORPHEME, "존재?:", os.path.exists(MORPHEME))
    print(">> KEYPOINT:", KEYPOINT, "존재?:", os.path.exists(KEYPOINT))
    video_map = scan_video_dirs(KEYPOINT)
    print(">> 현재 풀린 keypoint 클립 폴더 개수:", len(video_map))
    warn_if_word_tree(video_map)

    morpheme_files = find_morpheme_files(MORPHEME)
    print(">> morpheme JSON 개수:", len(morpheme_files))

    print(">> 문장 메타 스캔 중...")
    meta, skipped, no_korean = scan_sentence_meta(morpheme_files, video_map)
    print(">> 현재 매칭되는 문장 클립:", len(meta),
          " / 파싱 스킵:", skipped,
          " / 한국어키 없음(글로스 대체):", no_korean)

    catalog = load_json(catalog_path, {})
    processed = set(load_json(processed_path, []))
    print(">> 이미 처리한 문장 클립:", len(processed), "  catalog:", len(catalog))

    if len(catalog) == 0 and len(processed) > 0:
        raise SystemExit(
            ">> 중단: catalog 는 비었는데 processed 기록만 있습니다.\n"
            "   dataset_sen_3d 를 정리한 뒤 다시 수집하세요."
        )

    print(">> 새 문장 클립 로딩 중...")
    added = 0
    failed = 0
    bytes_written = 0

    for i, item in enumerate(meta, 1):
        name = item["name"]
        if name in processed:
            continue

        video_dir = video_map[name]
        x_raw, x_norm = load_sentence_sequence(video_dir, item["start"], item["end"])
        if x_raw is None:
            failed += 1
            continue

        T = int(x_raw.shape[0])
        need = x_raw.nbytes + (x_norm.nbytes if SAVE_NORM else 0)
        if added == 0:
            # 첫 클립 기준으로 대략 여유만 확인 (가변 길이라 정확하진 않음)
            check_free_space(OUT_DIR, need * max(8, min(32, len(meta) - len(processed))))

        raw_path = os.path.join(clip_dir, f"{name}.npy")
        save_npy_atomic(raw_path, x_raw)
        rec = {
            "name": name,
            "korean": item["korean"],
            "gloss": item["gloss"],
            "start": item["start"],
            "end": item["end"],
            "duration": round(item["end"] - item["start"], 3),
            "T": T,
            "angle": item["angle"],
            "group": item["group"],
            "raw": f"clips/{name}.npy",
            "shape": list(x_raw.shape),
            "korean_from_meta": item["korean_from_meta"],
        }
        if SAVE_NORM:
            norm_path = os.path.join(norm_dir, f"{name}.npy")
            save_npy_atomic(norm_path, x_norm)
            rec["norm"] = f"clips_norm/{name}.npy"
            rec["norm_shape"] = list(x_norm.shape)

        catalog[name] = rec
        processed.add(name)
        added += 1
        bytes_written += need

        # 중간 저장: 끊겨도 이어서 하게
        if added % 50 == 0:
            save_json_atomic(catalog_path, catalog)
            save_json_atomic(processed_path, sorted(processed))
            print(f"   ...추가 {added}  (최근 T={T}  {item['korean'][:30]})")

    save_json_atomic(catalog_path, catalog)
    save_json_atomic(processed_path, sorted(processed))

    kor_counter = Counter(v["korean"] for v in catalog.values())
    gloss_len = Counter(len(v["gloss"]) for v in catalog.values())
    t_vals = [v["T"] for v in catalog.values()] or [0]
    stats = {
        "clips": len(catalog),
        "unique_sentences": len(kor_counter),
        "T_min": int(min(t_vals)),
        "T_max": int(max(t_vals)),
        "T_mean": round(float(np.mean(t_vals)), 1),
        "gloss_len": dict(sorted(gloss_len.items())),
        "top_sentences": kor_counter.most_common(20),
        "last_added": added,
        "last_failed": failed,
    }
    save_json_atomic(stats_path, stats)

    print(">> 이번에 추가된 문장 클립:", added, "  실패:", failed)
    print(">> 누적 문장 클립:", len(catalog),
          "  서로 다른 문장:", len(kor_counter))
    print(">> 길이(프레임): min", stats["T_min"],
          " max", stats["T_max"], " mean", stats["T_mean"])
    print(">> 저장:", os.path.abspath(OUT_DIR))
    print(">> 문장 상위 10:")
    for s, c in kor_counter.most_common(10):
        print(f"     {c:4d}  {s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--peek", action="store_true",
        help="수집하지 않고 SEN morpheme 키 구조만 출력",
    )
    args = parser.parse_args()
    if args.peek:
        peek_morpheme(MORPHEME, n=2)
    else:
        main()
