# Coding AI Agent

DeepAgents CLI 기반 Coding AI Agent. 장기 메모리, 동적 SubAgent, 모델 Fallback, WebUI를 지원합니다.

**오픈소스 모델 우선 사용** — OpenRouter를 통해 Qwen, Nemotron, GLM 등 오픈소스 모델을 활용합니다.

## Architecture

```
┌─────────────── WebUI (Streamlit) ────────────────┐
│  Chat │ Memory Dashboard │ SubAgent │ Settings    │
└───────────────────┬──────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────┐
│         DeepAgents CLI (create_cli_agent)          │
│  Built-in: Filesystem, Shell, SubAgent,           │
│            Skills, Summarization, Memory           │
├──────────────────────────────────────────────────┤
│              Custom Extensions                     │
│  ModelFallbackMiddleware   (OpenRouter → Ollama)  │
│  LongTermMemoryMiddleware  (ChromaDB vector)      │
│  SubAgentLifecycleMiddleware (동적 생성/소멸)       │
│  AgentLoopGuard            (무한루프 방어)          │
└──────────────────────────────────────────────────┘
```

## Features

| 기능 | 설명 |
|------|------|
| **DeepAgents CLI 통합** | create_cli_agent() 기반, 파일시스템/셸/스킬/메모리 내장 |
| **장기 메모리** | ChromaDB 벡터 스토어, 4개 카테고리 (domain_knowledge, user_preferences, code_patterns, project_context) |
| **동적 SubAgent** | 런타임 생성/실행/소멸, Registry 상태 관리, 5가지 타입 (code_writer, researcher, reviewer, debugger, general) |
| **모델 Fallback** | OpenRouter 오픈소스 모델 → Ollama 로컬 LLM, Circuit Breaker 패턴 |
| **Agentic Loop 방어** | Max iterations (25회), empty response guard (3회), stuck detection (동일 호출 3회) |
| **WebUI** | Streamlit 4페이지: Chat (분할패널), Memory 대시보드, SubAgent 모니터, Settings |

## Quick Start

### 요구사항

- **Python 3.11+** (DeepAgents CLI 요구)
- OpenRouter API Key (오픈소스 모델용) 또는 Ollama (로컬 모델용)

### 로컬 실행

```bash
# 1. 의존성 설치
pip install -e .

# Ollama 지원이 필요하면:
pip install -e ".[ollama]"

# 2. .env 파일 생성
cp .env.example .env
# OPENROUTER_API_KEY를 설정하세요

# 3. WebUI 모드
python -m coding_agent --webui

# 4. CLI 모드
python -m coding_agent

# 5. 디버그 모드
python -m coding_agent --debug
```

### Docker (권장)

```bash
# 1. .env 파일 생성
cp .env.example .env
# OPENROUTER_API_KEY를 설정하세요

# 2. 실행
docker compose up --build

# 3. 브라우저에서 접속
# http://localhost:8501

# Ollama 로컬 LLM도 함께 실행하려면:
docker compose --profile with-ollama up --build
```

## Model Priority (오픈소스 모델)

기본 모델 우선순위 (OpenRouter 오픈소스):

1. `qwen/qwen3.5-35b-a3b` — Qwen 3.5 (추천)
2. `nvidia/nemotron-3-super-120b-a12b` — NVIDIA Nemotron (추천)
3. `z-ai/glm-5v-turbo` — GLM-5v (추천)
4. `deepseek/deepseek-chat-v3-0324` — DeepSeek V3
5. `qwen/qwen-2.5-coder-32b-instruct` — Qwen 2.5 Coder
6. **Fallback**: Ollama `qwen2.5-coder:7b` (로컬)

모델 실패 시 자동으로 다음 모델로 전환됩니다.
Circuit Breaker: 3회 연속 실패 시 5분간 해당 모델 스킵.

> **Note**: 프론티어 클로즈드 모델 (Claude Opus 4.6, GPT 5.4, Gemini 3.1 Pro) 은 사용하지 않습니다.

## 개발 요구사항 체크리스트

- [x] **DeepAgents CLI 활용**: `create_cli_agent()` 기반 에이전트 조립
- [x] **장기 메모리 / 지식 저장 체계**
  - [x] ChromaDB 벡터 스토어 (시맨틱 검색)
  - [x] 4개 카테고리: 도메인 지식, 사용자 선호, 코드 패턴, 프로젝트 컨텍스트
  - [x] 개인화: user_preferences 카테고리로 개발자 맞춤 기억
  - [x] 도메인 지식 저장: domain_knowledge 카테고리
  - [x] 프로젝트 맞춤화: project_context 카테고리
  - [x] DeepAgents AGENTS.md 네이티브 메모리도 병행
- [x] **동적 SubAgent 호출**
  - [x] 런타임 생성 (spawn) → 실행 (running) → 소멸 (cleanup)
  - [x] SubAgentRegistry로 상태 관리
  - [x] 이벤트 타임라인 (WebUI 실시간 모니터링)
  - [x] 동시 실행 수 제한 (기본 3개)
- [x] **Agentic Loop 방어**
  - [x] Max iterations (25회) 초과 시 자동 종료
  - [x] Empty response 연속 감지 (3회)
  - [x] Stuck detection: 동일 도구+인자 3회 반복 감지
  - [x] 모델 실패 시: Circuit Breaker → 다음 모델 전환 → Ollama 최종 폴백
- [x] **오픈소스 모델 사용**: 추천 모델 3개 + 추가 2개 포함

## Testing

### WebUI 테스트 (사이드바 Test Prompts)

| 테스트 | 프롬프트 | 확인 사항 |
|--------|----------|-----------|
| **SubAgent** | `Analyze by spawning sub-agents...` | 화면 자동 분할, SubAgent 생성/실행/소멸 |
| **Memory** | `Remember that I prefer Python type hints...` | memory_store → memory_search 호출 |
| **Multi-Agent** | `spawn a code_writer... then a reviewer...` | 순차 SubAgent 생성 |
| **Fallback** | `Write a simple hello world in Python` | 정상 응답, 모델명 표시 |

### CLI 테스트

```bash
python -m coding_agent

You> /status     # 모델별 Circuit Breaker 상태
You> /memory     # 메모리 카테고리별 통계
You> /subagents  # SubAgent 실행 이력
You> /quit       # 종료
```

## Project Structure

```
src/coding_agent/
├── __main__.py              # CLI 엔트리포인트
├── agent.py                 # DeepAgents CLI create_cli_agent() 래핑 + AgentLoopGuard
├── config.py                # 설정 관리 (오픈소스 모델 우선순위)
├── middleware/
│   ├── model_fallback.py    # OpenRouter→Ollama + CircuitBreaker (AgentMiddleware)
│   ├── long_term_memory.py  # ChromaDB 벡터 메모리 (AgentMiddleware)
│   └── subagent_lifecycle.py # 동적 SubAgent 관리 (도구 + Registry)
├── memory/
│   ├── store.py             # ChromaDB PersistentClient 벡터 스토어
│   └── categories.py        # 메모리 카테고리 Enum
└── webui/
    ├── app.py               # Streamlit 메인 앱
    └── pages/
        ├── chat.py          # 분할 패널 Chat UI
        ├── memory.py        # Memory 대시보드
        ├── subagents.py     # SubAgent 모니터
        └── settings.py      # 설정 페이지
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API 키 | (required) |
| `OLLAMA_BASE_URL` | Ollama 서버 URL | `http://localhost:11434` |
| `LOCAL_FALLBACK_MODEL` | 로컬 fallback 모델 | `qwen2.5-coder:7b` |
| `MEMORY_DIR` | 메모리 저장 경로 | `~/.coding_agent/memory` |
| `MAX_SUBAGENTS` | 최대 동시 SubAgent 수 | `3` |
