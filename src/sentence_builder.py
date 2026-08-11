# -*- coding: utf-8 -*-
"""
수어 단어열 -> 자연스러운 한국어 문장 변환 모듈
하이브리드 방식: 규칙(RULE_DICT) 우선 -> LLM 보완 -> 폴백(단순 연결)

LLM 백엔드는 교체 가능하도록 추상화되어 있음:
  - DummyLLM  : LLM 없이 폴백만 사용 (오프라인/테스트)
  - APILLM    : 네이버 CLOVA Studio HyperCLOVA X API 호출
  - LocalLLM  : (추후) 로컬 모델용 자리
"""

import os
import json
import requests


# =========================================================
# 1) 규칙 기반 사전 (데모 즉시 동작 보장 - LLM 불필요)
#    자주 나오는 단어 조합은 여기서 확정 문장으로 바로 변환
#    key: 단어 튜플(정렬 X, 인식된 순서 그대로), value: 완성 문장
# =========================================================
RULE_DICT = {
    ("머리", "아프다"): "머리가 아파요.",
    ("배", "아프다"): "배가 아파요.",
    ("병원", "가다"): "병원에 가고 싶어요.",
    ("진료", "예약", "원하다"): "진료 예약을 하고 싶어요.",
    ("신분증", "여기", "있다"): "신분증 여기 있어요.",
    ("계좌", "만들다", "원하다"): "계좌를 만들고 싶어요.",
    ("돈", "찾다", "원하다"): "돈을 찾고 싶어요.",
    ("얼마",): "얼마예요?",
    ("도와주다",): "도와주세요.",
    ("감사하다",): "감사합니다.",
    ("안녕하다",): "안녕하세요.",
}


def rule_based(words):
    """단어 튜플이 규칙 사전에 있으면 확정 문장 반환, 없으면 None."""
    key = tuple(words)
    return RULE_DICT.get(key)


def fallback(words):
    """LLM도 규칙도 실패했을 때: 단어를 그냥 이어붙이기."""
    if not words:
        return ""
    return " ".join(words) + "."


# =========================================================
# 2) LLM 백엔드 추상화
#    calling code는 to_sentence(words)만 호출하면 됨.
#    나중에 로컬 모델로 교체해도 호출부는 그대로.
# =========================================================
class LLMBackend:
    def to_sentence(self, words):
        """단어 리스트 -> 문장(str). 실패 시 None 반환."""
        raise NotImplementedError


class DummyLLM(LLMBackend):
    """LLM 없이 항상 폴백을 쓰게 하는 더미(테스트/오프라인용)."""
    def to_sentence(self, words):
        return None


class APILLM(LLMBackend):
    """
    네이버 CLOVA Studio HyperCLOVA X API 백엔드.
    - 엔드포인트: https://clovastudio.stream.ntruss.com/v3/chat-completions/{model}
    - 인증: Authorization: Bearer {API Key}
    - 데모/변환 용도에는 경량 모델 HCX-DASH-002 권장(빠르고 저렴).
    """

    ENDPOINT = "https://clovastudio.stream.ntruss.com/v3/chat-completions/{model}"

    SYSTEM_PROMPT = (
        "너는 한국 수어 단어열을 자연스러운 한국어 문장으로 바꾸는 번역기다.\n"
        "규칙:\n"
        "1. 입력으로 주어진 단어들만 사용해 하나의 자연스러운 문장을 만든다.\n"
        "2. 입력에 없는 새로운 정보나 내용을 절대 지어내지 않는다.\n"
        "3. 병원/관공서/은행 상황에 맞는 정중한 존댓말로 만든다.\n"
        "4. 설명 없이 완성된 문장 하나만 출력한다.\n"
        "예시: 입력 [머리, 아프다] -> 출력: 머리가 아파요."
    )

    def __init__(self, api_key=None, model="HCX-DASH-002",
                 request_id=None, timeout=10):
        # API 키는 환경변수 CLOVA_API_KEY 로 두는 것을 권장(코드에 하드코딩 금지)
        self.api_key = api_key or os.environ.get("CLOVA_API_KEY", "")
        self.model = model
        self.request_id = request_id  # 선택: 요청 추적용 UUID
        self.timeout = timeout

    def to_sentence(self, words):
        if not self.api_key:
            print("[APILLM] CLOVA_API_KEY 가 설정되지 않았습니다. 폴백으로 넘어갑니다.")
            return None

        url = self.ENDPOINT.format(model=self.model)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.request_id:
            headers["X-NCP-CLOVASTUDIO-REQUEST-ID"] = self.request_id

        user_text = "입력 단어: [" + ", ".join(words) + "]\n문장:"

        body = {
            "messages": [
                {"role": "system",
                 "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
                {"role": "user",
                 "content": [{"type": "text", "text": user_text}]},
            ],
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 100,
            "temperature": 0.3,       # 지어내기 방지: 낮게
            "repetitionPenalty": 1.1,
            "stop": [],
        }

        try:
            # 스트리밍 안 쓰고 한 번에 받기 (Accept: application/json 기본)
            resp = requests.post(url, headers=headers,
                                 data=json.dumps(body), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            # 응답 상태 확인
            status = data.get("status", {})
            if status.get("code") not in ("20000", None):
                print(f"[APILLM] API 오류: {status}")
                return None

            content = data["result"]["message"]["content"]
            return content.strip() if content else None

        except requests.exceptions.RequestException as e:
            print(f"[APILLM] 요청 실패: {e}")
            return None
        except (KeyError, ValueError) as e:
            print(f"[APILLM] 응답 파싱 실패: {e}")
            return None


class LocalLLM(LLMBackend):
    """
    (추후) 로컬 모델용 자리.
    병원/은행처럼 개인정보 민감 환경에서 온프레미스로 돌릴 때 사용.
    지금은 미구현 - 인터페이스만 맞춰둠.
    """
    def __init__(self, model_name="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"):
        self.model_name = model_name
        self._pipe = None  # 실제 로드는 필요 시점에

    def to_sentence(self, words):
        # TODO: transformers pipeline 으로 구현
        return None


# =========================================================
# 3) 통합 진입점
#    규칙 -> LLM -> 폴백 순서로 문장 생성
# =========================================================
def build_sentence(words, llm=None):
    """
    words: 인식된 수어 단어 리스트, 예) ["머리", "아프다"]
    llm  : LLMBackend 인스턴스 (None이면 규칙+폴백만 사용)
    return: 완성된 한국어 문장(str)
    """
    words = [w for w in words if w and w != "-"]  # 빈 값/무효 라벨 제거
    if not words:
        return ""

    # 1단계: 규칙 사전
    s = rule_based(words)
    if s:
        return s

    # 2단계: LLM 보완
    if llm is not None:
        s = llm.to_sentence(words)
        if s:
            return s

    # 3단계: 폴백
    return fallback(words)


# =========================================================
# 4) 단독 테스트
#    python sentence_builder.py
# =========================================================
if __name__ == "__main__":
    tests = [
        ["머리", "아프다"],              # 규칙 히트
        ["신분증", "여기", "있다"],       # 규칙 히트
        ["병원", "예약", "머리", "아프다"],  # 규칙 미스 -> LLM/폴백
    ]

    print("=== 규칙 + 폴백만 (LLM 없음) ===")
    for t in tests:
        print(f"  {t}  ->  {build_sentence(t)}")

    print("\n=== 규칙 + APILLM (환경변수 CLOVA_API_KEY 필요) ===")
    api_llm = APILLM(model="HCX-DASH-002")
    for t in tests:
        print(f"  {t}  ->  {build_sentence(t, llm=api_llm)}")
