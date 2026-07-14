---
title: SIGIR26' - GIGP+, A CPU-GPU Co-Processing Engine for Multi-Vector Retrieval
author: blakewoo
date: 2026-7-14 22:00:00 +0900
categories: [Paper]
tags: [Paper, Vector Database] 
render_with_liquid: false
use_math: true
---

# GIGP+: A CPU-GPU Co-Processing Engine for Multi-Vector Retrieval

> ※ 본 논문을 읽기 전에 [IGP](https://blakewoo.github.io/posts/IGP-SIGIR25'/) 논문을 읽고 오는게 좋다.
{: .prompt-tip }

## 1. 문제 정의
PLAID-GPU는 GPU의 병렬성을 잘 살릴 수 있지만, 불필요한 수많은 후보군을 선정하기 때문에 Re-rank 과정에서 많은 연산 시간을 필요하게 한다.
IGP는 좀 더 유망한 후보군을 소수 뽑지만 GPU를 병렬성을 활용하지 못한다.   
이를 해결하기 위핸 GIGP+ 논문이 등장했다고 말하고 있다.

## 2. GIGP+ 디자인
### 2.1. Index build
GIGP+가 Index를 빌드하는 절차를 크게 보자면 아래와 같다.

```
1. 문서 몇개를 샘플링하여 Centroid를 생성
2. 모든 문서를 Multi-vector로 변환 및 Centroid에 할당 및 잔차 압축
3. IVF 파일 생성 : Centroid -> 문서 ID
4. IVF offset/index를 GPU에서 접근하기 좋은 contiguous array로 저장
5. score group을 만들고, group 위치를 IVF 안에 기록
```

이전의 논문인 IGP와 다른점은 Centroid들에 문서 Vector의 ID가 할당되는 것이 아닌 PLAID와 같이 문서의 ID가 할당되었다는 점과
Centroid들로 ip-NSW를 만들지 않고, GPU에서 IVF 파일을 한번에 가져오기에 용이한 형태로 저장한다는 점이다.

### 2.2. Search
전체적인 검색 절차는 아래와 같다.

```
1. query vector q와 centroid C의 score q^T C 계산 및 score 높은 centroid들을 K개 선택
2. 선택된 centroid들의 IVF list를 병렬 fetch
3. (doc_id, <c, q>) tuple array T[q] 생성
4. score group별로 tuple을 재배열
5. GPU kernel에서 document-query별 AtomicMax
6. query vector별 max score를 합산해 document score 생성
7. top-φref candidate document 선택
8. candidate ID를 host로 보내 VQ/SQ code fetch
9. candidate document vector를 복원
10. full MaxSim rerank
```

위 절차를 세세하게 하나씩 설명해보겠다.

#### 1) query vector q와 centroid C의 score q^T C 계산 및 score 높은 centroid들을 선택
Query를 Multi-vector로 인코딩한다. 그러면 Query는 다수의 벡터가 된다. 이 벡터 하나를 q라고 하겠다.   
GIGP+ 는 이전의 IGP와 동일하게 각 q에 대해서 비슷한 것들을 탐색하는 것이다.

각 q에 대해서 모든 Centroid들과의 score를 구하고 내림차순으로 정렬한다. 이후 K개 만큼 선정한다.   

#### 2) 선택된 centroid들의 IVF list를 병렬 fetch
선택된 centroid들의 IVF list 길이를 이용해 offset array Δ를 계산한다.
여기서 offset array Δ란 아래와 같다.

아래와 같은 Centroid가 있다고 할때
```
c(1), c(2), c(3)
```

각 IVF LIST의 길이가 아래와 같다면
```
|IVF[c(1)]| = 3
|IVF[c(2)]| = 5
|IVF[c(3)]| = 2
```

각 Cetroid와 Query가 score를 문서에 대해서 표현(A array로 표현)하기 위해서는 아래와 같은 길이가 필요하다.
```
A length = 3 + 5 + 2 = 10
```

이를 표현하는 방식은 각 Centroid가 1차원 배열의 어디부터 시작하는지 offset을 적어두는 것으로 아래와 같이 표현된다.
```
Δ = [0, 3, 8]
```

Δ를 shared memory에 cache함으로써 빠른 접근이 가능하게 한다.

이후 각 centroid c(i)에 대해 IVF[c(i)]를 병렬로 읽은 뒤에 tuple array A에 저장한다.

```
A[Δ[i] + j] = (IVF[c(i)][j], <c(i), q>)
```

이후 A의 앞 n개를 반환한다.

#### 3) (doc_id, <c, q>) tuple array T[q] 생성
query는 여러 vector로 구성되므로, PrepareTupleArray(Q, φcand)가 각 query vector마다 2)번 절차를 수행한다.   
이후 결과는 아래와 같은 tuple로 나타난다.

```
T[q1] = [(doc_id, <c, q1>), ...]
T[q2] = [(doc_id, <c, q2>), ...]
...
T[qm] = [(doc_id, <c, qm>), ...]
```

#### 4) score group별로 tuple을 재배열
#### 5) GPU kernel에서 document-query별 AtomicMax
#### 6) query vector별 max score를 합산해 document score 생성
#### 7) top-φref candidate document 선택
#### 8) 1차 후보군 잔차 복구 및 GPU Memory로 전송 및 Rerank

> ※ 추가 업데이트 예정이다.
{: .prompt-tip }

# 참고문헌
- Bian, Zheng, Man Lung, Yiu, and Bo, Tang. "GIGP+: A CPU-GPU Co-Processing Engine for Multi-Vector Retrieval." . In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 88–98). Association for Computing Machinery, 2026.







