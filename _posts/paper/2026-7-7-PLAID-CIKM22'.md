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

## 2. PLAID의 검색 파이프라인

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

### 2.1. Stage 1: Initial Candidate Generation
### 2.2. Stage 2: Centroid Interaction with Pruning
#### ※ Centroid Pruning
### 2.3. Stage 3: Centroid Interaction without Pruning
### 2.4. Stage 4: Final Ranking with Decompression

> ※ 추가 업데이트 예정이다.
{: .prompt-tip }

# 참고문헌
- Santhanam, Keshav, Omar, Khattab, Christopher, Potts, and Matei, Zaharia. "PLAID: An Efficient Engine for Late Interaction Retrieval." . In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (pp. 1747–1756). Association for Computing Machinery, 2022.






