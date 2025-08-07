---
title: langGraph - 개요
author: blakewoo
date: 2025-8-6 17:50:00 +0900
categories: [AI]
tags: [AI, langGraph]
render_with_liquid: false
---

# LangGraph
## 1. 개요
```
Trusted by companies shaping the future of agents – including Klarna, Replit, Elastic,
and more – LangGraph is a low-level orchestration framework for building, managing,
and deploying long-running, stateful agents.
```

LangGraph 공식 사이트에 올라와있는 LangGraph에 대한 설명이다.   
간단히 해석하자면 장기 실행, 상태 저장이 가능한 에이전트를 구축하고 관리하고 배포하기 위한 저수준 오케스트레이션 프레임 워크라고 한다.    

쉽게 설명해서 다수의 에이전트를 엮어서 하나의 서비스를 만들기 쉬운 플랫폼이라고 할 수 있다.

사용 예시)
- 복잡한 멀티 에이전트 시스템 : 각각의 전문화 된 에이전트를 다수로 운용함으로써 전문성과 성능을 높임
- 대화형 AI : 고객 지원 챗봇을 구성할 경우 다양한 분기를 사용함으로써 개인화되고 역동적인 상호작용이 가능
- 자동화 워크플로우 : 데이터 분석 파이프라인 구축간 각 노드가 데이터 정리, 변환, 집계등을 각각 담당하여 성능을 높일 수 있음

## 2. 구성 요소
이름이 LangGraph이니 만큼 Graph 형태이되 유향 비순환 그래프(Directed Acyclic Graph,DAG)의 형태이고
구성 요소 역시 Graph를 구성하는 컴포넌트들의 이름을 하고 있다.

### 1) Graph
전체 워크플로우를 표현하는 객체를 말한다. 노드 목록 및 edge 리스트를 포함하고 있다.

### 2) Node
그래프 안에서 수행되는 최소 단위 작업을 말한다. LLM 호출은 이 Node에서 이뤄지며 LLM 호출 뿐만아니라
외부 API 호출이나 데이터베이스 쿼리등의 작업등을 모두 포함한다.

기본적으로 Python 함수로 구현되며 현재 상태를 입력으로 받아서 처리하고 업데이트된 상태를 출력한다.

```python
def aging(state):
    return {"age": state["age"] + 1}

graph.add_node("aging", aging)
```
### 3) State
그래프의 전체 데이터 흐름을 관리하는 핵심 요소로 주로 TypedDict나 Pydantic BaseModel을 사용하여 정의한다.   
그래프의 모든 노드와 엣지에 대한 입력 스키마 역할을 한다.

```python
from typing import TypedDict

class CustomState(TypedDict):
    name: str
    age : int
```

### 4) Edge
Edge는 노드 간의 연결을 나타내는 것으로 그래프 실행 흐름을 결정한다.   
에지를 통해 노드에서 다음 노드로 상태가 전달되며 시작은 START, 끝은 END라는 특별한 상수로 나타낸다.

아래의 예시는 START에서 aging Node를 거쳐 END로 향하는 그래프이다.

```python
graph.add_edge(START, "aging")
graph.add_edge("aging", END)
```

여기서 aging은 앞서 Node 부분에서 정의한 Node이다.

### 5) checkpoint
앞서 개요에서 말했듯이 langGraph에서는 ```상태 저장```이 가능한 에이전트를 만들수있다.   
이 상태저장이 가능하기 위해서는 저장소가 필요한데, LangGraph에서는 기본적으로 inMemory 저장소를 지원하며
postgres와 sqlite를 기본으로 하는 DB 저장소를 제공한다.
만약 interface를 구현한다면 다른 저장소를 사용할 수도 있다.

## 3. LangChain과의 관계
동일한 곳에서 만들었다. 하지만 목적이 좀 다르다.    
아래의 표를 살펴보자.

| 구분                                  | LangChain                  | LangGraph                      |
| ----------------------------------- | -------------------------- | ------------------------------ |
| 추상화 수준                              | 체인(chain) 구성 중심, 고수준 인터페이스 | 그래프(graph) 오케스트레이션, 저수준 제어권 제공 |
| 상태 관리(state)                        | 기본적으로 stateless            | stateful 워크플로우 지원              |
| 반복/순환 로직                            | 따로 지원하지 않음                 | 사이클(cycle)·조건부 분기·이벤트 기반 로직 내장 |
| 사람 개입 지원                            | 제한적                        | 그래프 단계마다 휴먼-인-더-루프 통합 가능       |

간단히 말해서 좀 더 세밀한 제어와 조율을 원한다면 , Stateful한 워크 플로우를 AI Agent를 통해
개발하고자한다면 LangGraph를 사용하는게 맞다.

> ※ 추가 업데이트 및 검증 예정
{: .prompt-tip }

# 참고자료
- [LangGraph 가이드북 - 에이전트 RAG with 랭그래프](https://wikidocs.net/261590)
- [공식 LangGraph 사이트](https://langchain-ai.github.io/langgraph/)
- Chat GPT의 도움 약간(문구 수정과 글 구성등)
- https://academy.langchain.com/courses/intro-to-langgraph "Foundation: Introduction to LangGraph - LangChain Academy"
- https://pypi.org/project/langgraph/0.0.25/ "langgraph - PyPI"
- [LangGraph 예시](https://www.linkedin.com/pulse/exploring-frontiers-ai-top-5-use-cases-langchain-dileep-kumar-pandiya-hos3e/)
- [LangGraph - Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
