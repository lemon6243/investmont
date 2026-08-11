# -*- coding: utf-8 -*-
"""
text_to_gloss.py
한국어 문장 -> 수어 글로스(단어 순서열) 변환.
모듈 7(sentence_builder)의 LLM 연동을 '반대 방향'으로 재활용.

핵심 제약: 우리가 실제로 보유한 단어(available_words) 안에서만 고르게 한다.
          없는 단어를 지어내면 재생이 불가능하므로.
"""

import os
import json
import requests


# ------------------------------------------------------------
# 1) 규칙 사전 (자주 쓰는 문장은 확정 글로스로 - 안정성 보장)
#    key: 입력 문장(정규화), value: 글로스 리스트
# ------------------------------------------------------------
RULE_GLOSS = {
    "머리가 아파요": ["머리", "아프다"],
    "배가 아파요": ["배", "아프다"],
    "병원에 가고 싶어요": ["병원", "가다", "원하다"],
    "도와주세요": ["도와주다"],
    "얼마예요": ["얼마"],
}


def _normalize(text):
    return text.strip().replace("?", "").replace(".", "").replace("!", "")


def rule_gloss(text):
    return RULE_GLOSS.get(_normalize(text))


# ------------------------------------------------------------
# 2) LLM 백엔드 (CLOVA HyperCLOVA X)
# ------------------------------------------------------------
class GlossLLM:
    ENDPOINT = "https://clovastudio.stream.ntruss.com/v3/chat-completions/{model}"

    def __init__(self, api_key=None, model="HCX-DASH-002", timeout=10):
        self.api_key = api_key or os.environ.get("CLOVA_API_KEY", "")
        self.model = model
        self.timeout = timeout

    def to_gloss(self, text, available_words):
        """text -> 글로스 리스트. 실패 시 None."""
        if not self.api_key:
            print("[GlossLLM] CLOVA_API_KEY 없음 -> 규칙/폴백만 사용")
            return None

        # 보유 단어 목록을 프롬프트에 넣어 그 안에서만 고르게 강제
        word_list = ", ".join(sorted(available_words))
        system = (
            "너는 한국어 문장을 한국수어(KSL) 글로스 순서로 바꾸는 변환기다.\n"
            "규칙:\n"
            "1. 반드시 아래 '사용 가능 단어' 목록에 있는 단어만 사용한다.\n"
            "2. 목록에 없는 단어는 절대 만들지 않는다. 표현 못 하면 가능한 단어만 낸다.\n"
            "3. 조사·어미는 빼고, 수어 어순(주로 주어-목적어-동사)으로 배열한다.\n"
            "4. 결과는 JSON 배열 하나만 출력한다. 예: [\"머리\", \"아프다\"]\n"
            f"사용 가능 단어: {word_list}"
        )
        user = f"문장: {text}\n글로스 JSON 배열:"

        body = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "text", "text": user}]},
            ],
            "topP": 0.8, "topK": 0, "maxTokens": 100,
            "temperature": 0.2, "repetitionPenalty": 1.1, "stop": [],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = self.ENDPOINT.format(model=self.model)

        try:
            resp = requests.post(url, headers=headers,
                                 data=json.dumps(body), timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json()["result"]["message"]["content"].strip()
            # JSON 배열만 뽑아내기 (모델이 앞뒤에 말 붙일 수 있으므로)
            s, e = content.find("["), content.rfind("]")
            if s == -1 or e == -1:
                print(f"[GlossLLM] 배열 파싱 실패: {content}")
                return None
            gloss = json.loads(content[s:e+1])
            # 보유 단어로 한번 더 필터 (안전장치)
            gloss = [g for g in gloss if g in available_words]
            return gloss if gloss else None
        except Exception as ex:
            print(f"[GlossLLM] 오류: {ex}")
            return None


# ------------------------------------------------------------
# 3) 통합 진입점: 규칙 -> LLM -> 폴백(보유단어 매칭)
# ------------------------------------------------------------
def text_to_gloss(text, available_words, llm=None):
    # 1) 규칙
    g = rule_gloss(text)
    if g:
        return [w for w in g if w in available_words]  # 보유한 것만

    # 2) LLM
    if llm is not None:
        g = llm.to_gloss(text, available_words)
        if g:
            return g

    # 3) 폴백: 문장에 등장하는 보유 단어를 순서대로 줍기
    found = []
    for w in available_words:
        if w in text:
            found.append((text.find(w), w))
    found.sort()
    return [w for _, w in found]


if __name__ == "__main__":
    # 단독 테스트 (보유 단어 예시)
    avail = {"머리", "아프다", "병원", "가다", "원하다", "얼마", "도와주다"}
    llm = GlossLLM()
    tests = ["머리가 아파요", "병원에 가고 싶은데 도와주세요", "얼마예요?"]
    for t in tests:
        print(f"{t}  ->  {text_to_gloss(t, avail, llm=llm)}")
