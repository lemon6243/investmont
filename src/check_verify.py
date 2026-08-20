# check_verify.py
# rots/*.npz 검증: (1) 첫 프레임 identity  (2) 프레임 간 연속성
#
# 사용법:
#   python check_verify.py            # 전체 단어 훑어서 문제 있는 것만 요약
#   python check_verify.py 머리       # 특정 단어 하나 자세히
import os
import sys
import glob
import numpy as np

ROTS_DIR = os.path.join("dataset_all_3d", "rots")

# 팔 6개 본 (데모 핵심) — 이것들만 집중 검사
ARM_BONES = [
    "upperarm_r", "lowerarm_r", "hand_r",
    "upperarm_l", "lowerarm_l", "hand_l",
]

IDENTITY = np.array([0, 0, 0, 1], np.float32)
SIM_THRESH = 0.9        # 인접 프레임 유사도 이 아래면 '튐'
IDENT_THRESH = 0.98     # 첫 프레임이 identity와 이만큼은 닮아야 정상


def quat_sim(a, b):
    """부호 무시한 쿼터니언 유사도 (1=동일)."""
    return abs(float(np.dot(a, b)))


def check_bone(q):
    """q: (T,4). (첫프레임 identity 여부, 최소 인접 유사도, 튄 프레임 리스트) 반환."""
    T = len(q)
    ident_sim = quat_sim(q[0], IDENTITY)
    worst = 1.0
    jumps = []
    for t in range(1, T):
        s = quat_sim(q[t], q[t - 1])
        if s < worst:
            worst = s
        if s < SIM_THRESH:
            jumps.append((t, round(s, 3)))
    return ident_sim, worst, jumps


def check_file(path, verbose=False):
    """한 npz 검사. 문제 요약 dict 반환."""
    data = np.load(path, allow_pickle=True)
    word = os.path.splitext(os.path.basename(path))[0]
    problems = {"ident": [], "jump": []}

    for bone in ARM_BONES:
        if bone not in data:
            continue
        q = data[bone].astype(np.float32)
        if len(q) < 2:
            continue
        ident_sim, worst, jumps = check_bone(q)

        if ident_sim < IDENT_THRESH:
            problems["ident"].append((bone, round(ident_sim, 3)))
        if jumps:
            problems["jump"].append((bone, worst, jumps))

        if verbose:
            flag = ""
            if ident_sim < IDENT_THRESH:
                flag += "  [첫프레임 non-identity]"
            if jumps:
                flag += f"  [튐 {len(jumps)}곳]"
            print(f"  {bone:12s} 첫프레임유사도={ident_sim:.3f} "
                  f"최소인접유사도={worst:.3f}{flag}")
            if jumps and verbose:
                for t, s in jumps[:5]:
                    print(f"      f{t}: 유사도={s}")

    has_problem = bool(problems["ident"] or problems["jump"])
    return word, has_problem, problems


def main():
    args = sys.argv[1:]

    if args:
        # 특정 단어 하나 자세히
        word = args[0]
        path = os.path.join(ROTS_DIR, f"{word}.npz")
        if not os.path.exists(path):
            print(f"파일 없음: {path}")
            return
        print(f"[{word}] 팔 6개 본 검사")
        check_file(path, verbose=True)
        return

    # 전체 훑기
    files = sorted(glob.glob(os.path.join(ROTS_DIR, "*.npz")))
    print(f">> {len(files)}개 단어 검사 중...")

    ident_bad = []
    jump_bad = []
    for path in files:
        word, has_problem, problems = check_file(path)
        if problems["ident"]:
            ident_bad.append((word, problems["ident"]))
        if problems["jump"]:
            jump_bad.append((word, problems["jump"]))

    print(f"\n===== 결과 =====")
    print(f"전체: {len(files)}개")
    print(f"첫 프레임이 identity 아님: {len(ident_bad)}개")
    print(f"프레임 간 튐 있음: {len(jump_bad)}개")

    if ident_bad:
        print(f"\n[첫 프레임 non-identity 상위 10개]")
        for word, bones in ident_bad[:10]:
            names = ", ".join(f"{b}({s})" for b, s in bones)
            print(f"  {word}: {names}")

    if jump_bad:
        print(f"\n[프레임 튐 상위 10개]")
        for word, info in jump_bad[:10]:
            summary = ", ".join(f"{b}(최소{w:.2f},{len(j)}곳)" for b, w, j in info)
            print(f"  {word}: {summary}")

    if not ident_bad and not jump_bad:
        print("\n모든 단어 정상. 파이썬 회전 계산 검증 통과.")


if __name__ == "__main__":
    main()
