---
title: Google 논문 - MUVERA : Multi-Vector Retrieval via Fixed Dimensional Encodings 분석
author: blakewoo
date: 2025-10-23 23:30:00 +0900
categories: [Paper]
tags: [Paper, Google, VectorDB, LSH, Multi-vector] 
render_with_liquid: false
use_math: true
---

# Muvera
## 1. 문제 정의 
기존의 Vector Search에서 말하는 Vector는 단일 벡터에 대한 내용이었다.   
어떤 Document를 단일 벡터로 만들면 아무래도 의미 손실이 많이 일어난다. 이 때문에 ColBERT 계열과 같은 임베딩 기법이 나왔다.
ColBERT 계열은 쿼리/문서를 토큰별 여러 임베딩(멀티벡터)으로 표현하는데 이렇게 표현하면 품질이 좋지만, 검색(특히 후보 검색)이 매우 비싸다.   
왜냐면 문서 하나에 다수의 벡터가 나오기 때문에 이를 각기 다 계산해주어야하기 때문이다.   
이를 문제로 본 google research 팀은 솔루션으로 Muvera를 제시했다.
MUVERA는 이 멀티벡터 유사도(Chamfer / MaxSim)를 단일 벡터(inner-product) 검색으로 근사하여
기존의 고속 MIPS 인덱스(예: DiskANN)를 그대로 쓸 수 있게 하는 방식이다.

## 2. 핵심 아이디어
각 쿼리/문서의 멀티 벡터를 단일 고정 차원 벡터로 변환하여 이 두 벡터의 내적이 Chamfer similarity를 근사하도록 설계하는 것이다.

> ※ 추가 업데이트 및 검증 예정
{: .prompt-tip }
### 1) 공간 분할
SimHash(또는 k-means)로 latent space를 B개 cluster로 나눈다.
이렇게 나눌 경우 가까운 포인트가 같은 cluster에 들어갈 확률이 높다.

### 2) 블록 구성
각 cluster k에 대해 쿼리 블록은 그 cluster에 속한 쿼리 벡터들의 합, 문서 블록은 그 cluster가 속한
문서 벡터들의 평균으로 둔다. 이렇게 각 Cluster마다 d차원의 블록을 만든다.

### 3) 내부 및 최종 투영
각 블록을 낮은 차원으로 랜덤 프로젝션으로 줄여서 전체 차원을 조절하고 반복들을 독립적으로 생성하여 붙인다.

### 4) 빈 클러스터 처리
어떤 cluster에 그 문서의 토큰이 하나도 없으면 0으로 두지 않고, 그 cluster에
'가장 가까운' 문서 토큰을 대신 넣어 근사 실패(빈 충돌)를 줄인다.(문서 FDE에만 적용)

이렇게 만들어지면 데이터 불변(data-oblivious)한 변환으로 멀티벡터 유사도를 단일 내적으로 근사하게 된다.

> ※ 추가 업데이트 및 검증 예정
{: .prompt-tip }
## 3. 실험 결과
FDE 검색은 기존 SV 히어리스틱(PLAID 기반)보다 후보 수를 훨씬 적게 조회하면서 동일하거나 더 나은 Recall을 달성함(예: 같은 recall을 위해 2–5× 적은 후보).
End-to-end(BEIR 데이터셋들 기준): 평균 Recall 약 +10%, 평균 Latency 약 −90%(대부분 데이터셋에서 빠름)를 보고.
MS MARCO에서는 같은 수준(또는 약간 못할 수 있음)으로 나왔음(PLAID가 MS MARCO에 특화 튜닝된 탓일 가능성).
PQ(예: PQ-256-8)를 쓰면 메모리 32× 감소, QPS는 유지 혹은 개선되며 recall 손실은 거의 없음.

> ※ 추가 업데이트 및 검증 예정
{: .prompt-tip }

# 참고문헌
- Laxman Dhulipala, , Majid Hadian, Rajesh Jayaram, Jason Lee, and Vahab Mirrokni. "MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings." (2024).


