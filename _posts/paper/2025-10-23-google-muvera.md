---
title: Google 논문 - MUVERA : Multi-Vector Retrieval via Fixed Dimensional Encodings 분석
author: blakewoo
date: 2025-10-24 23:30:00 +0900
categories: [Paper]
tags: [Paper, Google, VectorDB, LSH, Multi-vector] 
render_with_liquid: false
use_math: true
---

# Muvera
## 1. 문제 정의 
기존의 Vector Search에서 말하는 Vector는 단일 벡터를 다루는 일이 많았다.   
여기서 말하는 단일 벡터란, 한 개의 Document를 한 개의 임베딩 벡터로 만드는 것을 말한다.   
Document 값이 얼마나 크든 간에 내용이 어떻든 상관없이 한 개의 벡터로 만들어버리므로 의미 손실이 일어난다.

이 때문에 Multi-vertor 기법이 나왔는데 이는 한 개의 Document를 토큰, 문장, 섹션등 여러 벡터로 표현해서 쿼리의 특정 부분과 문서의 특정 부분을  
직접 매칭 할 수 있기 때문에 정밀도가 올라간다. 이 기법 중에 가장 대표적인 것이 바로 ColBERT이다.

방금 언급했듯이 ColBERT 계열은 쿼리/문서를 토큰별 여러 임베딩(멀티벡터)으로 표현하는데 이렇게 표현하면 품질이 좋지만, 검색(특히 후보 검색)은 
시간이 많이 들고 복잡한 연산이 되어버린다. 왜냐면 문서 하나에 다수의 벡터가 나오기 때문에 이를 각기 다 계산해주어야하기 때문이다.   
각기 계산해준다는게 좀 안 와닿을 수 있는데 아래와 같은 수식이라고 생각하면 된다.

각 쿼리 토큰을 $q \in Q$ 라고하면 모든 문서 토큰 $p \in P$ 에 대해서 가장 유사한 것 $\max_{p \in P} \langle q, p \rangle$ 을 구한다.
그리고 이들을 합산하여 chamfer Similarity를 계산하는데 쿼리 토큰수 $|Q|$와 문서 토큰수 $|P|$에 비례한다.
이를 전체 문서 집합 n에 대해서 수행하면 아래와 같은 양의 연산량이 나온다.

$$ O(|Q| \cdot \max_i |P_i| \cdot n) $$

위와 같이 계산도 복잡할 뿐더러 이전에 전통적인 Single Vector Search 방식도 사용하지 못한다.
따라서 이를 문제로 본 google research 팀은 솔루션으로 Muvera를 제시했다.
MUVERA는 이 멀티벡터 유사도(Chamfer / MaxSim)를 단일 벡터(inner-product) 검색으로 근사하여
기존의 고속 MIPS 인덱스(예: DiskANN)를 그대로 쓸 수 있게 하는 방식이다.

## 2. 핵심 아이디어
각 쿼리/문서의 멀티 벡터를 단일 고정 차원 벡터로 변환하여 이 두 벡터의 내적이 Chamfer similarity를 근사하도록 설계하는 것이다.

> ※ 추가 업데이트 및 검증 예정
{: .prompt-tip }
### 1) 공간 분할
SimHash(또는 k-means)로 latent space를 B개 cluster로 나눈다.
이렇게 나눌 경우 가까운 포인트가 같은 cluster에 들어갈 확률이 높다.

#### ※ SimHash
simHash 고차원(문서·문장 등)의 유사도를 빠르게 근사하기 위해 각 특징(feature)을 해시해서 가중 합한 뒤 부호(sign)만 남기는 방식을 말하며
LSH(지역 민감 해시)이다.
결과는 이진 fingerprint(예: 64비트)이고, 이진 벡터 사이의 해밍거리(Hamming distance)가 유사도를 나타낸다.

#### # Hamming distance
해밍 거리란 두 값이 얼마나 차이가 있냐는 나타내는 거리이다.   
어떤 이진수 11001101과 10111011가 있다고 할 때 이 두 값의 해밍 거리는 아래와 같이 구할 수 있다.

```
11001101
10111011 XOR
------------
01110110 
```

XOR 한 값의 1이 5개이므로 해밍거리는 5이다.

> ※ 추가 업데이트 및 검증 예정
{: .prompt-tip }
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
- Khattab, Omar, and Matei, Zaharia. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." . In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 39–48). Association for Computing Machinery, 2020.



