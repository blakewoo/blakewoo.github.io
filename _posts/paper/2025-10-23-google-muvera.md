---
title: Google 논문 - MUVERA, Multi-Vector Retrieval via Fixed Dimensional Encodings 분석
author: blakewoo
date: 2025-10-25 23:30:00 +0900
categories: [Paper]
tags: [Paper, Google, VectorDB, LSH, Multi-vector] 
render_with_liquid: false
use_math: true
---

# Muvera
## 1. 문제 정의 
Neural embedding은 클러스터링 및 분류와 같은 많은 작업 중에서 정보 검색의 핵심 도구가 되었다. 이는 어떤 문서를 임베딩 벡터화 했을 때 
다수의 문서 중에서 가장 거리가 가까운 값일 수록 해당 문서에 가깝다는 것이 알려졌기 때문이다. 이러한 원리를 이용하여 벡터 데이터를 Search하는
방법들이 많이 등장했고 발달하고 있다.

기존의 Vector Search에서 말하는 Vector는 단일 벡터를 다루는 일이 많았다.   
여기서 말하는 단일 벡터란, 한 개의 Document를 한 개의 임베딩 벡터로 만드는 것을 말한다.   
Document 값이 얼마나 크든 간에 내용이 어떻든 상관없이 한 개의 벡터로 만들어버리므로 의미 손실이 일어난다.   
가령 컴퓨터의 역사에 대해서 논하는 Document가 있다고 해보자, 컴퓨터의 역사를 서술했지만 하드웨어 위주로 조명하여 서술한 Document라고 할때
하드웨어에 대한 내용은 소실되고 컴퓨터의 역사라는 것만 남는 식이다.   
사용자는 하드웨어 관점의 컴퓨터 역사를 알고 싶다고 할 때 위와 같이 소실된 형태의 벡터가 Document 나타낸다면 찾아내지 못할 가능성이 있다.

이 때문에 Multi-vertor 기법이 나왔는데 이는 한 개의 Document를 토큰, 문장, 섹션등 여러 벡터로 표현해서 쿼리의 특정 부분과 문서의 특정 부분을 
직접 매칭 할 수 있기 때문에 정밀도가 올라간다. 이 기법 중에 가장 대표적인 것이 바로 ColBERT이다.

방금 언급했듯이 ColBERT 계열은 쿼리/문서를 토큰별 여러 임베딩(멀티벡터)으로 표현하는데 이렇게 표현하면 각 문맥에 대해서
의미 손실이 적어져 검색 품질은 좋아지지만 시간이 많이 들고 복잡한 연산이 되어버린다. 왜냐면 문서 하나에 다수의 벡터가 나오기
때문에 쿼리 역시 여러 임베딩으로 만들어 이를 각기 계산해주어야하기 때문이다. (이 연산을 MaxSim 혹은 Chamfer Similarity라고 한다)   
또한 기존의 1개의 Document에서 1개의 임베딩 벡터가 나왔던것과 달리 다수의 임베딩 벡터가 생성하기 때문에 용량도 커지게 된다.

특히 이 Chamfer Similarity 방식은 이전에 전통적인 Single Vector Search 방식을 사용하지 못한다.
따라서 이를 문제로 본 google research 팀은 솔루션으로 Muvera를 제시했다.
MUVERA는 이 멀티벡터 유사도(Chamfer / MaxSim)를 단일 벡터(inner-product) 검색으로 근사하여
기존의 고속 MIPS(Maximum inner product search) 인덱스(예: DiskANN)를 그대로 쓸 수 있게 하는 방식이다.

### ※ MaxSim or Chamfer similarity
쿼리의 토큰 임베딩 집합 $Q$ 와 문서의 토큰 임베딩 집합 $P$ 가 있다고 할 때 각 집합의 원소를 아래와 같이 정의할 수 있다.   
$$ Q = \{q_1, q_2, \dots, q_{|Q|}\} $$   
$$ P = \{p_1, p_2, \dots, p_{|P|}\} $$

여기서 $ q_i, p_j \in R^d $ 이며 d는 임베딩 차원의 벡터이다.

어떤 쿼리와 문서의 토큰 임베딩인 $q$ 와 $p$ 에 대해서 내적을 구하는데
어떤 쿼리 $q$ 에 대해서 문서의 모든 토큰 임베딩에 대해서 내적을 구한다. 

그러면 총 문서의 토큰 임베딩 수 만큼의 내적이 나오는데, 그 중에서 가장 최고 값을 찾는다.
$$ \max_{p \in P} \langle q, p \rangle $$ 

이는 특정 쿼리 토큰에 대해 문서 내에서 가장 잘 매칭되는 토큰을 찾는 과정이다.    
이 과정을 모든 쿼리 임베딩 토큰에 대해서 반복하고 찾을 최고 값들을 모두 더한다.

$$ \max_{p \in P} \langle q_1, p \rangle, \max_{p \in P} \langle q_2, p \rangle, \dots, \max_{p \in P} \langle q_{|Q|}, p \rangle = CHAMFER(Q, P) = \sum_{q \in Q} \left( \max_{p \in P} \langle q, p \rangle \right) $$

위 계산은 한 개의 Document에 대해서만 이뤄진 것이다. 만약 가장 가까운 Document를 찾고 싶다면 모든 문서에 대해서 Chamfer similarity를 구해서 그 중에 
가장 값이 높은 것을 찾으면 될 것이다.

## 2. 핵심 아이디어
각 쿼리/문서의 멀티 벡터를 고정 차원 싱글 벡터로 변환하여 이 두 벡터의 내적이 Chamfer similarity를 근사하도록 설계하는 것이다.   

### 1) 공간 분할
한 개의 문서에 다수의 임베딩 벡터가 있다. 이를 SimHash로 latent space를 B개 cluster로 나눈다.
이렇게 나눌 경우 가까운 포인트가 같은 cluster에 들어갈 확률이 높다.
SimHash로 나눌 경우 B는 아래의 값이 된다.

$$ B = 2^{K_{sim}} $$

K-means로 나눌 수도 있지만 SimHash가 데이터에 구애받지 않기 때문에 좀 더 우수하다고 한다.   

위 함수를 $ \phi $ 라고 하면 이 함수는 d차원의 실수를 B로 사상시킨다.

$$ \phi: R^d \rightarrow [B] $$

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

### 2) 블록 구성
각 cluster k에 대해 쿼리 블록은 그 cluster에 속한 쿼리 벡터들의 합,

$$ q^{(k)} &= \sum_{q \in Q, \phi(q)=k} q \\ $$

문서 블록은 그 cluster가 속한 문서 벡터들의 평균으로 둔다.   

$$ p^{(k)} &= \frac{1}{|P \cap \phi^{-1}(k)|} \sum_{p \in P, \phi(p)=k} p$$

이렇게 각 Cluster마다 d차원의 블록을 만든다.      
이 BLOCK들을 다 붙여서 임시 FDE를 만든다.   
임시 FDE의 차원은 블록 개수가 B이고, 차원이 d라고 한다면 $Bd$ 로 표현할 수 있다.

단, 어떤 cluster에 그 문서의 토큰이 하나도 없으면 0으로 두지 않고, 그 cluster에
'가장 가까운' 문서 토큰을 대신 넣어 근사 실패(빈 충돌)를 줄인다.    
이러한 작업은 문서 FDE에만 적용하며 쿼리 FDE에는 적용하지 않는다.

이렇게 만들어지면 데이터 불변(data-oblivious)한 변환으로 멀티벡터 유사도를 단일 내적으로 근사하게 된다.

### 3) 내부 및 최종 투영
임시 FDE는 블록을 단순히 붙여서 차원이 매우 높다. 차원이 높으면 계산하는데 문제가 생기므로 각 블록의 차원을 적절하게 줄이는 과정이 필요하다.   
이를 위해서는 투영 함수가 필요한데, 논문에서는 투영 함수를 아래와 같이 정의한다.

$$ \psi(x) = \left( \frac{1}{\sqrt{d_{proj}}} \right) Sx $$   

여기서 x는 투영될 원래의 블록 벡터 $ q^{(k)} $ 또는 $ p^{(k)} $ 의 값이다.

$$ d_{proj} $$ 값은 값을 실험적으로 튜닝하는 경우가 많으며 논문에서는 8, 16, 32값으로 세팅했다.
위와 같이 투영 함수를 적용하여 차원이 작아진 블럭을 붙여서 최종 FDE로 만든다.

> ※ 추가 업데이트 및 검증 예정
{: .prompt-tip }

## 3. 실험 결과
FDE 검색은 기존 Single Vector 휴리스틱(PLAID 기반)보다 후보 수를 훨씬 적게 조회하면서 동일하거나 더 나은 Recall을 달성함(예: 같은 recall을 위해 2–5× 적은 후보).
End-to-end(BEIR 데이터셋들 기준): 평균 Recall 약 +10%, 평균 Latency 약 −90%(대부분 데이터셋에서 빠름)를 보고.
MS MARCO에서는 같은 수준(또는 약간 못할 수 있음)으로 나왔음(PLAID가 MS MARCO에 특화 튜닝된 탓일 가능성).
PQ(예: PQ-256-8)를 쓰면 메모리 32× 감소, QPS는 유지 혹은 개선되며 recall 손실은 거의 없음.

> ※ 추가 업데이트 및 검증 예정
{: .prompt-tip }

# 참고문헌
- Laxman Dhulipala, , Majid Hadian, Rajesh Jayaram, Jason Lee, and Vahab Mirrokni. "MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings." (2024).
- Khattab, Omar, and Matei, Zaharia. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." . In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 39–48). Association for Computing Machinery, 2020.



