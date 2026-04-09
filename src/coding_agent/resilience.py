"""Explicit resilience policy table and helpers for agentic loop failures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FailurePolicy:
    failure_type: str
    detect_signal: str
    max_retries: int
    fallback: str
    user_status: str
    safe_stop_condition: str


POLICIES: dict[str, FailurePolicy] = {
    "model_timeout": FailurePolicy(
        failure_type="model_timeout",
        detect_signal="LLM response timeout or interrupted stream",
        max_retries=1,
        fallback="retry once, then switch model via fallback middleware",
        user_status="현재 모델 응답 지연으로 재시도 중",
        safe_stop_condition="retry and fallback both exhausted",
    ),
    "no_progress_loop": FailurePolicy(
        failure_type="no_progress_loop",
        detect_signal="same action or no state change across repeated iterations",
        max_retries=1,
        fallback="shrink plan or stop async expansion",
        user_status="진전 없음 감지, 전략 전환 시도",
        safe_stop_condition="still no progress after one strategy change",
    ),
    "tool_call_error": FailurePolicy(
        failure_type="tool_call_error",
        detect_signal="invalid tool schema, missing args, or execution error",
        max_retries=1,
        fallback="rewrite tool call once",
        user_status="도구 호출 형식 오류 수정 중",
        safe_stop_condition="same tool failure repeats after rewrite",
    ),
    "subagent_failure": FailurePolicy(
        failure_type="subagent_failure",
        detect_signal="subagent enters failed or blocked state",
        max_retries=1,
        fallback="replace with alternate role or stop and report",
        user_status="하위 작업 실패, 대체 경로 시도",
        safe_stop_condition="alternate path also fails",
    ),
    "external_api_error": FailurePolicy(
        failure_type="external_api_error",
        detect_signal="4xx/5xx, rate limit, network failure",
        max_retries=2,
        fallback="use cached/local path or safe stop",
        user_status="외부 서비스 오류 대응 중",
        safe_stop_condition="error persists after bounded retries",
    ),
    "safe_stop": FailurePolicy(
        failure_type="safe_stop",
        detect_signal="insufficient evidence, risky action, or missing permission",
        max_retries=0,
        fallback="none",
        user_status="안전하게 중단됨. 사용자 확인 필요",
        safe_stop_condition="immediate",
    ),
}


def get_policy(failure_type: str) -> FailurePolicy:
    return POLICIES[failure_type]
