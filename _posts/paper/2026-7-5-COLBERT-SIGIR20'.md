---
title: SIGIR20' - COLBERT, Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
author: blakewoo
date: 2026-7-5 22:00:00 +0900
categories: [Paper]
tags: [Paper, Vector Database] 
render_with_liquid: false
use_math: true
---

# COLBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
멀티 벡터 검색의 시초에 가까운 논문이다. 생각해보니 벡터 데이터 베이스 관련 포스팅을 해두고 이러한 기념비적인
논문을 리뷰하지 않았다는 점이 이상하다고 생각이 들어서 해당 논문을 리뷰하게 되었다.

## 1. 문제 정의
BM25 같은 전통 검색은 빠르다. 하지만 단어가 정확히 겹치지 않으면 검색 성능이 떨어진다.
예를 들어 사용자가 “car repair cost” 라고 검색했는데 문서에는  “automobile maintenance expense”   
라고 되어 있으면 의미는 비슷하지만 단어가 다르기 때문에 잘 못 찾을 수 있다.
이걸 논문에서는 vocabulary mismatch, 즉 어휘 불일치 문제라고 한다.

이러한 문제를 해결하기 위해 Bert를 이용해서 검색하는 방식 역시 등장했다.
BERT 기반 랭커는 쿼리와 문서를 함께 입력한다. (아래 참조)

```
[CLS] query [SEP] document [SEP] → BERT 모델 → relevance score
```

이 방식은 성능은 좋지만 매우 느리다. 문서 1000개를 reranking하려면 쿼리 하나에 대해 BERT를 1000번 실행해야 한다.
논문에서는 BERT 기반 랭커가 기존 모델보다 계산량이 수백~수천 배 커질 수 있다.
ColBERT는 이 문제를 해결하기 위해 등장했다.

## 2. Late-Interaction 아키텍처
사실 이 논문의 핵심 디자인이자, 현재 Multi-vector 검색의 헤게모니를 쥐고 있는 방식이다.   

### 2.1 Query Encoder
쿼리는 BERT WordPiece 토큰으로 나눈다.
그리고 쿼리 앞에는 [Q]라는 특수 토큰을 붙인다.
```[CLS] [Q] query tokens ...```
논문에서는 쿼리 길이를 고정하기 위해 부족한 부분을 [MASK] 토큰으로 채운다.
이것을 query augmentation이라고 부른다.

예를 들어 실제 쿼리가 짧으면 다음과 같이 된다.

```
[CLS] [Q] what is colbert [MASK] [MASK] ...
```

이 [MASK] 위치도 BERT를 통과하면서 쿼리 문맥에 맞는 표현을 만들 수 있다.
논문에서는 이것이 일종의 부드러운 query expansion 역할을 한다고 설명한다.

### 2.2 Document Encoder
문서도 BERT로 인코딩한다.

문서 앞에는 [D]라는 특수 토큰을 

```
[CLS] [D] document tokens ...
```

문서에는 [MASK]를 추가하지 않는다.
또한 punctuation, 즉 문장부호에 해당하는 벡터는 제거한다.
문장부호 벡터는 검색 품질에 큰 도움이 되지 않으면서 저장 공간만 차지한다고 보기 때문이다.

### 2.3 Linear Layer와 벡터 차원 축소
BERT의 기본 hidden dimension은 보통 768이다.
하지만 ColBERT는 문서의 모든 토큰 벡터를 저장해야 하므로 768차원 그대로 저장하면 공간이 너무 크다.
그래서 BERT 출력 뒤에 linear layer를 붙여서 더 작은 차원으로 줄인다.
논문 실험에서는 보통 128차원을 사용한다.

```
BERT output 768-dim → Linear layer → 128-dim
```

그 후 벡터를 L2 normalize한다. 이렇게 하면 dot product가 cosine similarity와 같아진다.

아래와 같은 두 가지 방식으로 나뉜다.

### 2.4.1. BM25 + Rerank
전통적인 BM25로 1차 후보군 1000개를 뽑고, 후보군과 쿼리의 Maxsim을 구하여 Rerank하는 방법이다.

#### 2.4.2. IVF + Rerank
Faiss와 같은 라이브러리에서 제공하는 [IVF](https://blakewoo.github.io/posts/VectorDB-ANNs/) 로 전체 문서의 Vector를 Index로 만들어서 
1차 후보군 1000개를 뽑은 뒤 후보군과 쿼리의 Maxsim을 구하여 Rerank하는 방법이다.


> ※ Rerank 방식에 대한 세부 내용은 이전에 포스팅한 [이곳](https://blakewoo.github.io/posts/VectorDB-Rerank/) 을 참고하라.
{: .prompt-tip }


# 참고문헌
- Omar Khattab, , and Matei Zaharia. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." (2020).


