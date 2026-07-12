---
title: Vector DB - Rerank
author: blakewoo
date: 2025-11-4 23:00:00 +0900
categories: [Database]
tags: [Database, DBMS ,VectorDB, Rerank]
render_with_liquid: false
use_math: true
---

# Vector DB - Re-rank
## 1. 개요
이전에 ANN(approximate nearest neighbor)에 대해서 포스팅한 적이 있다.   
하지만 이 ANN의 경우 대부분 Single vector retrieval을 대상으로 하는 방식이다.

여기서 말하는 Single vector retrieval이란 검색 대상인 Document들은 각각 하나의 vector로 변환되어서 적재되어있고
Query 역시 Single vector로 바뀌어 앞서 포스팅했던 ANN 방법대로 검색하는 것이다.

하지만 이렇게 Document를 한 개의 Vector로 바꾸니 의미 손실이 일어났다. 그래서 나온게 Multi vector retrieval 방식이다.
가장 처음 언급한 [ColBERT](https://blakewoo.github.io/posts/COLBERT-SIGIR20'/) 라는 논문이다. 해당 논문에서 제시한 방식은 2가지이다.
BM25라는 전통적인 검색 시스템이 상위 K를 뽑은 뒤 MaxSim(Max similarity)를 구하여 나온 score 대로 재 순위화 하거나
Faiss와 같은 라이브러리에서 제공하는 IVF 방식으로 1차 후보군을 뽑은 뒤 score 대로 재 순위화를 하는 것이다.

위와 같이 1차 후보군 + 재순위화하는 방식을 late interaction이라고 하는데, 이게 현재 Multi vector retrieval의 전형적인 방식이 되었으며
BM25 방식을 사용하던 1차 후보 추출기를 다른 Single vector 검색 방식으로 대체하거나 다른 알고리즘을 사용하는 방식으로 여러가지 Variation 이 생겼지만
기본적인 Rerank 방식이 달라지진 않았다.

## 2. COLBERT에서 언급했던 Rerank시 사용한 알고리즘들
최초의 late interaction 논문인 COLBERT에서 L2 거리와 Cosine 유사도로 Reranking한 내용이 나와서 

### L2 Distance(유클리디안 거리, Euclidean Distance)

유클리디안 거리로 xy좌표로 피타고라스 공식으로 거리 구하듯이 구하는 방식이다.

어떤 다차원 벡터 $P,Q$ 가 아래와 같은 형태의 벡터라고 하자

$$ P = {p_{1},p_{2},p_{3},...p_{i}}$$   

$$ Q = {q_{1},q_{1},q_{1},...q_{i}}$$   

$$ d(P,Q)=\sqrt{\sum_{i=1}^{n}(p_{i}-q_{i})^{2}} $$   

위와 같은 수식으로 거리를 구할 수 있다. 

### Cosine 유사도
코사인 유사도(Cosine Similarity)는 두 벡터 간의 사잇각( $\theta$ )의 코사인 값을 이용하여 방향성 유사도를 측정하는 지표이다.
-1에서 1 사이의 값을 가지며, 1에 가까울수록 두 벡터의 방향이 같아 유사하고, 0이면 직교(무관계), -1이면 반대 방향을 의미한다.
벡터의 크기(길이)보다 방향의 패턴을 중요시할 때 주로 사용된다.

어떤 다차원 벡터 $P,Q$ 가 아래와 같은 형태의 벡터라고 하자

$$ P = {p_{1},p_{2},p_{3},...p_{i}}$$

$$ Q = {q_{1},q_{1},q_{1},...q_{i}}$$

코사인 유사도는 아래의 수식으로 구할 수 있다.

$$Cosine Similarity = \frac{P\cdot Q}{\left\| P \right\| \left\| Q \right\|} = \frac{\sum_{i=1}^{n}p_{i}q_{i}}{\sqrt{\sum_{i=1}^{n}p_{i}^{2}}\sqrt{\sum_{i=1}^{n}q_{i}^{2}}}  $$

### ※ 내적 기반 Maxsim을 사용하는 이유
일단 내적 자체가 GPU에 가장 친화적인 연산이다. 보면 알겠지만 L2나 Cosine 유사도는 좀 더 복잡하다.   
그리고 애당초 COLBERT자체가 모든 출력 임베딩의 Norm 값이 1이 되도록 정규화했기 때문에 내적을 취하더라도 코사인 유사도와 동일하다.

## 3. Maximum similarity
Google에서 나온 MUVERA 논문에서는 Chamfer similarity라고 하기도 했지만 ColBERT 논문에서는 Maximum similarity라고 했으니
Maximum Similarity 라고 하겠다(이하 Maxsim)
Rerank 방법에 대해서 이해하려면 Rerank를 하는 기준인 Maxsim score를 구하는 방법에 대해서 알아야한다.
어떤 Query 하나 Embedding vector 집합을 $ Q $, 어떤 Document 하나 Embedding vector 집합을 $ D $ 라고 할때
Maxsim score 를 구하는 공식은 아래와 같이 표현할 수 있다.

$$ score(Q,D) = \sum_{q\in Q}max_{d\in D}\left\langle q,d \right\rangle $$

Q의 Embedding vector 중 하나인 q와 D의 Embedding vector 중 하나인 d의 내적을 구했을 때 가장 큰 것(Max)을 찾는다.
각 q에 대해서 해당 D내에 가장 내적이 큰 d를 찾아서 구한 내적 값을 모두 더하면 score값이다. 

이 score를 어떤 Query에 대해서 모든 Document에 대해서 구했을 때 가장 Score가 높은 Document가 Query에 가장 가까운 문서라고 할 수 있다.

## 4. 예시
벡터 검색하는데 내적을 모르는 사람이 없을거라고는 생각하지만 이런 포스팅을 할 때는 읽는 이가 아무것도 모르고 읽는다고 가정해야한다고 했다.   
따라서 예시를 들어서 설명하겠다.

어떤 Document A~C를 Multi vector화 하니 아래와 같이 나왔다고 해보자.   
A = [{3,4},{4,2},{1,1}]   
B = [{2,1},{2,5},{9,9}]      
C = [{5,6},{5,2},{3,3}]  

Query Q를 벡터화하니 아래와 같이 나왔다.

Q = [{5,5},{8,4},{6,4}]

각각의 Embedding Vector 집합인데, 각각 순서에 대해 아래 첨자로 표현할 수 있다. ex) $Q_{1}$    
이 경우 각 Document A~C가 포함한 3개의 2차원 벡터 3개에 대해서 아래와 같은 연산을 할 수 있다.   

### 1) Q와 A의 Maxim Score
(5,5)(3,4)$^{T}$ = 35    
(5,5)(4,2)$^{T}$ = 30    
(5,5)(1,1)$^{T}$ = 10

최대값 35

(8,4)(3,4)$^{T}$ = 40    
(8,4)(4,2)$^{T}$ = 40    
(8,4)(1,1)$^{T}$ = 12 

최대값 40

(6,4)(3,4)$^{T}$ = 34    
(6,4)(4,2)$^{T}$ = 32    
(6,4)(1,1)$^{T}$ = 10

최대값 34

A의 MAXSIM = 109

### 2) Q와 B의 Maxim Score
(5,5)(2,1)$^{T}$ = 15   
(5,5)(2,5)$^{T}$ = 35   
(5,5)(9,9)$^{T}$ = 90

최대값 90

(8,4)(2,1)$^{T}$ = 20   
(8,4)(2,5)$^{T}$ = 36   
(8,4)(9,9)$^{T}$ = 108

최대값 108

(6,4)(2,1)$^{T}$ = 16   
(6,4)(2,5)$^{T}$ = 21   
(6,4)(9,9)$^{T}$ = 90

최대값 90

B의 MAXSIM = 288

### 3) Q와 C의 Maxim Score
(5,5)(5,6)$^{T}$ = 55   
(5,5)(5,2)$^{T}$ = 35   
(5,5)(3,3)$^{T}$ = 30

최대값 55

(8,4)(5,6)$^{T}$ = 64   
(8,4)(5,2)$^{T}$ = 48   
(8,4)(3,3)$^{T}$ = 36

최대값 64

(6,4)(5,6)$^{T}$ = 54   
(6,4)(5,2)$^{T}$ = 38   
(6,4)(3,3)$^{T}$ = 30

최대값 54

C의 MAXSIM = 173

### 4) 최종 결과
가장 가까운 문서는 B의 Score가 가장 크니 B가 가장 Query와 가깝다.

> ※ 추가 업데이트 및 검증 예정이고, 아직 완벽하지 않으니 참고만 바란다.
{: .prompt-tip }


# 참고자료
- Keshav Santhanam, , Omar Khattab, Christopher Potts, and Matei Zaharia. "PLAID: An Efficient Engine for Late Interaction Retrieval." (2022).
- Laxman Dhulipala, , Majid Hadian, Rajesh Jayaram, Jason Lee, and Vahab Mirrokni. "MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings." (2024).
- Khattab, Omar, and Matei, Zaharia. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." . In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 39–48). Association for Computing Machinery, 2020.



