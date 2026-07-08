---
title: CIKM22' - PLAID, An Efficient Engine for Late Interaction Retrieval
author: blakewoo
date: 2026-7-7 22:00:00 +0900
categories: [Paper]
tags: [Paper, Vector Database] 
render_with_liquid: false
use_math: true
---

# PLAID: An Efficient Engine for Late Interaction Retrieval
COLBERTv2 논문의 파생형이다. ColBERTv2와 동일한 저자가 좀 더 나은 형태의 방식을 제시한 것이다.      
기본적인 골자는 다르지 않다. 후보 문서를 많이 가져온 뒤에 각 후보 문서의 잔차를 복원하고 full maxsim으로 계산하는 기존 방식과는 달리    
Centroid ID 만으로 여러 단계의 비용이 싼 필터링을 수행한 뒤 남은 정말 유망한 후보군만 잔차 복원을 하여 Maxsim을 수행하는 방식이다.   

세부적인 방식은 아래와 같다.

## 1. 문제 정의
논문에서 이전의 ColBERTv2 검색에서 병목은 초기 후보군이 많으면 수만 개 문서의 압축 Vector를 가져와서 복원하고 계산하는 부분이
주요 병목이라고 말한다.
때문에 이 문제를 해결하기 위해 비용이 싼 필터링을 수행하여 후보군을 줄인다고 말한다.

## 2. PLAID의 Index 빌드
기본적으로는 ColBERTv2와 방식은 비슷하다.
```
Corpus passages
  ↓
[1] 문서 encoding
    ColBERTv2 encoder로 token-level embeddings 생성

  ↓
[2] Centroid selection
    sample token embeddings에 대해 k-means 수행
    centroid set C 생성

  ↓
[3] Vector quantization
    각 token embedding v를 nearest centroid C_t에 할당
    centroid ID 저장

  ↓
[4] Residual compression
    residual r = v - C_t 계산
    residual을 1-bit 또는 2-bit로 scalar quantization
    centroid ID + quantized residual 저장

  ↓
[5] PLAID-style inverted list 생성
    centroid → unique 문서 IDs

  ↓
[6] 문서별 centroid ID sequence 저장
    centroid interaction에서 사용할 lightweight bag-of-centroids representation

  ↓
[7] final rerank용 metadata 저장
    doclens, offsets, centroid IDs, residual codes
```

4번까지는 ColBERTv2와 동일하며 5번부터 다르다.   

### 1.PLAID 식 Inverted list 생성
기존 ColBERTv2에서는 Centroid ID에 문서의 토큰 ID가 달려있던 것과는 달리 PLAID에서는 아예 그 문서의 ID를 달아둔다.   
이렇게하면 이 문서의 ID가 문서의 토큰 ID를 기재하는 것보다 훨씬 ID 개수가 적기 때문에 IVF가 작아지기 때문에
용량 절감 효과가 크다.

### 2. 문서별 Centroid ID Sequence 저장
검색 과정에서 있을 Centroid와 문서의 상호작용에서 사용하기 위해서 문서가 어떤 Centroid를 갖고 있는지
ID를 저장해둔다.

### 3. final rerank용 metadata 저장
중간 단계가 조금 다르긴하지만 결국에는 전체 Maxsim 연산을 통해 Rerank를 하기 때문에
잔차 복구를 위한 정보가 필요하다. 따라서 아래의 정보도 저장한다.

```
  1. 각 token의 centroid ID
  2. 각 token의 quantized residual
  3. 문서별 token offset
  4. 문서별 token length
  5. centroid vectors
  6. residual bucket / lookup table
```

## 3. PLAID의 검색 파이프라인

PLAID의 큰 검색 파이프라인은 아래와 같다.

```
  Query encoding
    ↓
  query vector와 centroid 간 score 계산
    ↓
  centroid 기반으로 초기 candidate passage 생성
    ↓
  centroid interaction + pruning으로 1차 filtering
    ↓
  centroid interaction without pruning으로 2차 filtering
    ↓
  소수 candidate만 residual decompression
    ↓
  full MaxSim scoring
    ↓
  top-k 반환
```

위 과정을 하나씩 자세히 살펴보겠다.

### 2.1. Stage 1: Initial Candidate Generation
먼저 Query를 ColBERTv2 모델을 이용하여 Multi-vector화 한 뒤에 모든 Centroid들과
행렬 연산을 진행한다. 쿼리와 Centroid의 관련성 점수를 $S_{s,q}$ 라고 하고 Query 토큰의 행렬을 Q,
Centroid를 C라고 할때 아래와 같이 표현할 수 있다.

$$ S_{c,q} = C \cdot Q^{T}$$

검색시 설정한 nprobe만큼의 개수만큼 제일 큰 점수를 갖는 Centroid를 구한다.

### 2.2. Stage 2: Centroid Interaction with Pruning
대상 Query와 관련 있는 Centroid 후보군들을 구했다면 해당 Centroid에 엮여 있는 문서 ID를 가져온 뒤
해당 문서와 관련 있는 Centroid 값과 Query로 Maxsim 유사도 Score를 구하는데
앞서 Stage 1에서 Query와 Centroid들과의 유사도 $S_{c,q}$ 를 구해두었기 때문에 새로 연산을 할 필요없이
Score를 가져온다음에 Max만 취하면 된다. 구한 Score를 기반으로 ```ndocs```개 만큼 상위에서 자른다.

이렇게 문서를 문서 vector의 집합이 아닌 cetroid들의 집합으로 보고 Maxsim을 근사하는 것을 Centroid Interaction이라고 한다.

#### ※ Centroid Pruning
Stage2에서 말하는 Centroid Interaction을 하기 전에 중요하지 않은 Centroid를 먼저 제거하는데 이를 Centroid Pruning이라고 한다.   
Centroid와 Query Vector Token과의 최대 Score를 본다. 이 값이 threshold 값 $t_{cs}$ 보다 작으면 해당 Centroid를 제거한다.

여기서  $t_{cs}$ 은 사용자가 정하는 하이퍼 파라미터값으로 검색 성능을 보면서 사용자가 튜닝해야하는 값이다.   
논문에서는 각 하이퍼 파라미터에 대해 아래와 같이 설정하고 실험을 진행했다.

| 최종 검색 깊이 k | nprobe | t_cs | ndocs |
| ---------: | -----: | ---: | ----: |
|         10 |      1 |  0.5 |   256 |
|        100 |      2 | 0.45 |  1024 |
|       1000 |      4 |  0.4 |  4096 |


### 2.3. Stage 3: Centroid Interaction without Pruning
Stage 2에서 나온 ```ndocs``` 개에서 더 작은 후보군 개수 만큼 남기기 위해 Pruning 없이 Centroid Interaction을 진행한다.   
논문에서는 ```ndocs/4``` 개의 문서만 남기는게 가장 좋은 결과를 보였다고 말한다.

### 2.4. Stage 4: Final Ranking with Decompression
Stage 3에서 반환된 문서들을 모두 잔차 복구하여 Query와 Full maxsim 연산을 진행한 뒤 상위 K개 만큼 반환한다.


> ※ 추가 업데이트 예정이다.
{: .prompt-tip }

# 참고문헌
- Santhanam, Keshav, Omar, Khattab, Christopher, Potts, and Matei, Zaharia. "PLAID: An Efficient Engine for Late Interaction Retrieval." . In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (pp. 1747–1756). Association for Computing Machinery, 2022.






