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
가장 처음 언급한 ColBERT라는 논문이다. 해당 논문에서 제시한 방식은 간단하다. BM25라는 전통적인 검색 시스템이 상위 K를 뽑은 뒤
MaxSim(Max similarity)를 구하여 나온 score 대로 재 순위화 하는 것이다.

위와 같은 방식을 late interaction이라고 하는데, 이게 현재 Multi vector retrieval의 전형적인 방식이 되었으며
BM25 방식을 사용하던 1차 후보 추출기를 다른 Single vector 검색 방식으로 대체하거나 다른 알고리즘을 사용하는 방식으로 여러가지 Variation 이 생겼지만
기본적인 Rerank 방식이 달라지진 않았다.

## 2. Maximum similarity
Google에서 나온 MUVERA 논문에서는 Chamfer similarity라고 하기도 했지만 ColBERT 논문에서는 Maximum similarity라고 했으니
Maximum Similarity 라고 하겠다(이하 Maxsim)
Rerank 방법에 대해서 이해하려면 Rerank를 하는 기준인 Maxsim score를 구하는 방법에 대해서 알아야한다.
어떤 Query 하나 Embedding vector 집합을 $ Q $, 어떤 Document 하나 Embedding vector 집합을 $ D $ 라고 할때
Maxsim score 를 구하는 공식은 아래와 같이 표현할 수 있다.

$$ score(Q,D) = \sum_{q\in Q}max_{d\in D}\left\langle q,d \right\rangle $$

Q의 Embedding vector 중 하나인 q와 D의 Embedding vector 중 하나인 d의 내적을 구했을 때 가장 큰 것(Max)을 찾는다.
각 q에 대해서 해당 D내에 가장 내적이 큰 d를 찾아서 구한 내적 값을 모두 더하면 score값이다. 

이 score를 어떤 Query에 대해서 모든 Document에 대해서 구했을 때 가장 Score가 높은 Document가 Query에 가장 가까운 문서라고 할 수 있다.


> ※ 추가 업데이트 및 검증 예정이고, 아직 완벽하지 않으니 참고만 바란다.
{: .prompt-tip }


# 참고자료
- Keshav Santhanam, , Omar Khattab, Christopher Potts, and Matei Zaharia. "PLAID: An Efficient Engine for Late Interaction Retrieval." (2022).
- Laxman Dhulipala, , Majid Hadian, Rajesh Jayaram, Jason Lee, and Vahab Mirrokni. "MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings." (2024).
- Khattab, Omar, and Matei, Zaharia. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." . In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 39–48). Association for Computing Machinery, 2020.



