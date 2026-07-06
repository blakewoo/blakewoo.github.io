---
title: NAACL22' - COLBERTv2, Effective and Efficient Retrieval via Lightweight Late Interaction
author: blakewoo
date: 2026-7-6 22:00:00 +0900
categories: [Paper]
tags: [Paper, Vector Database] 
render_with_liquid: false
use_math: true
---

# COLBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction
COLBERT 논문의 파생형이다. 동일한 저자와 다른 공저자가 하나 더 붙어서 좀 더 나은 형태의 방식을 제시한 것이다.   
본래는 Model에 대한 개선과 LoTTE (Long-Tail Topic-stratified Evaluation for IR), 그러니까
지식 기반(Knowledge base) 외부에 존재하는 전문적이고 희소한(Long-tail) 주제들을 다루는 데이터셋까지
해당 논문의 기여에 포함이 되어있지만 이번 포스팅에서는 위 두 가지는 다루지 않고 넘어가겠다.

## 1. 문제 정의
기존 COLBERT 방식에 따르면 문서의 임베딩 된 사이즈가 너무 크다는 문제가 있다.   
1개의 문서를 표현 할때 128차원 벡터가 256개만 되더라도 float 기준 128KB가 된다.   
1000개의 문서만 되더라도 128000KB, 즉 128MB이다.    

이렇게 Index 구성을 위한 임베딩 문서의 용량이 너무 큰 문제를 해결하기 위해 아래와 같은 디자인을 이용하여
이 문제를 해결했다.

## 2. COLBERTv2에서의 Index 생성
ColBERTv2에서 주요하게 달라진 점이라고 하면 바로 잔차 표현(residual representation)을 사용한다는 점이다.   
이 방식은 아키텍처나 모델 훈련 방식의 변경 없이 기존 방식 그대로 용량이 줄어드는 방식이라고 논문에서 설명한다.   

### 2.1. Centroid 선택
문서의 임베딩 벡터 수의 제곱근에 비례하여 샘플링을 한뒤에 생성한 임베딩에 [K-Means]((https://blakewoo.github.io/posts/%EA%B8%B0%EA%B3%84%ED%95%99%EC%8A%B5-k-means/) ) 를 이용하여 Centroid 집합을 얻는다.

### 2.2. Corpus 인코딩
모든 문서를 임베딩 모델을 이용하여 다중 벡터로 만든다. 클러스터링 되어 얻은 Centroid들에 각 벡터들을 할당한다.
즉, 각 벡터의 대표값이 Centroid들이 되는 것이다. 하지만 이렇나 Centroid에서 원래 벡터들이 얼마나 떨어져 있는지
알 수 없으므로 Centroid에서 떨어진 값. 즉, 잔차까지 같이 포함하여 기재한다.

```
Centroid : c
Vector : v
잔차 : r
v = c + r 
```

이때 잔차를 표현할 때 몇 비트를 이용해서 표현할 것인가에 따라 Index 압축률이 달라지며, 자치하는 용량의 크기가 매우 달라진다.    
논문에서는 2bit만 이용해서 표현해도 recall의 손실이 거의 없었다고 주장한다.

### 2.3. IVF 생성
위와 같이 각 Centroid들에 배정된 Vector들을 대상으로 역인덱스 파일을 생성하여 디스크에 저장한다.

### ※ 잔차 표현법
Centroid와 그에 대해 차원당 몇 비트만을 이용해서 잔차를 기재하는 것까지는 그렇다치더라도 어떻게 잔차를 표현하는가?  
이는 [Product Quantization](https://blakewoo.github.io/posts/VectorDB-Product-Quantization/) 의 확장으로 설명이 될 수 있다고 한다.   
명확하게 논문에 나와있지는 않지만 논문에서 안내한 github repo에 올라와있는 구현체 코드를 읽어 봤을 때 2.1에서 샘플링되어 임베딩 된 Vector에서
구해진 잔차를 통해 bit 수에 따라(ex - 2bit시 4개 분할) 범위를 분할하여 코드북을 만드는 방식이다.

## 3. ColBERTv2에서의 검색
### 3.1. Centroid 탐색
쿼리 Q가 주어지면 쿼리의 Q의 각 벡터 $ q_{i} $ 에 대해 가장 가까운 nprobe 개의 중심을 찾는다.   

### 3.2. 잔차 압축 해제 및 Cosine 유사도 계산
Centroid에 할당된 문서 Vector들의 잔차 압축을 해제하고 $q_{i}$ 들과 코사인 유사도를 계산한다.   
이 과정에서 계산된 유사도 점수를 문서 ID별로 그룹화하여 쿼리 벡터마다 'MaxSim' 연산을 근사적으로 수행하여 Max-reduce(최대값 하나만 남기는 것)한다.
이러한 방법으로 후보 문서들을 선별한다. 이때의 유사도 합은 실제 MaxSim 점수의 하한선(lower bound) 역할을 한다.

### 3.3. Re-ranking
3.2. 에서 구한 후보 문서 ID에 속한 모든 잔차 압축된 벡터를 가지고와 잔차 압축을 해제한 뒤 Maxsim 연산을 통해 Rerank를 하여
최종 후보를 반환한다.

> ※ 추가 업데이트 예정이다.
{: .prompt-tip }

# 참고문헌
- Santhanam, Keshav, Omar, Khattab, Jon, Saad-Falcon, Christopher, Potts, and Matei, Zaharia. "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction." . In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 3715–3734). Association for Computational Linguistics, 2022.




