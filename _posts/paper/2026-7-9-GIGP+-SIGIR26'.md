---
title: SIGIR26' - GIGP+, A CPU-GPU Co-Processing Engine for Multi-Vector Retrieval
author: blakewoo
date: 2026-7-13 22:00:00 +0900
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
1. query vector q와 centroid C의 score q^T C 계산
2. score 높은 centroid들을 선택
3. 선택된 centroid들의 IVF list를 병렬 fetch
4. (doc_id, <c, q>) tuple array T[q] 생성
5. score group별로 tuple을 재배열
6. GPU kernel에서 document-query별 AtomicMax
7. query vector별 max score를 합산해 document score 생성
8. top-φref candidate document 선택
9. candidate ID를 host로 보내 VQ/SQ code fetch
10. candidate document vector를 복원
11. full MaxSim rerank
```

> ※ 추가 업데이트 예정이다.
{: .prompt-tip }

# 참고문헌
- Bian, Zheng, Man Lung, Yiu, and Bo, Tang. "GIGP+: A CPU-GPU Co-Processing Engine for Multi-Vector Retrieval." . In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 88–98). Association for Computing Machinery, 2026.







