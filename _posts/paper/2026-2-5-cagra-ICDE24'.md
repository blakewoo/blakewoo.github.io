---
title: ICDE24' - CAGRA, Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPU
author: blakewoo
date: 2026-2-5 22:00:00 +0900
categories: [Paper]
tags: [Paper, Vector Database] 
render_with_liquid: false
use_math: true
---

# CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPU
이전의 Graph 기반 Vector 검색은 CPU 기반이었다. 이번에 리뷰할 논문인 CAGRA는 NVIDIA에서 나왔는데
GPU를 이용해서 가속이 가능한 Graph 알고리즘이다. 이전에 다른 연구들이 있었지만 기존 Sorta 대비해서
훨씬 빠르다고 이야기하고 있다.

## 1. 개요
CAGRA는 대규모 데이터셋에 대한 Approximate Nearest Neighbor Search (ANNS)의 성능을 혁신하기 위해 NVIDIA GPU의 병렬 처리 능력과 메모리
대역폭을 최대한 활용하도록 설계된 최첨단 graph-based ANNS 방법론이다. 기존 ANNS 방법들은 GPU의 잠재력을 완전히 끌어내지 못했으며,
특히 graph construction 및 small-batch query 처리에서 병목 현상이 있었다. CAGRA는 이러한 격차를 해소하고,
CPU 및 GPU 기반의 기존 SOTA (State-of-the-Art) 구현체들을 능가하는 성능을 보여준다.

CAGRA의 핵심은 GPU 아키텍처에 최적화된 새로운 proximity graph 구조와 이를 구축하고 탐색하는 알고리즘에 있다.

## 2 CAGRA Graph 설계
CAGRA 그래프는 이론적인 그래프 특성보다는 실제 검색 구현 성능에 중점을 둔 휴리스틱(heuristic) 방식으로 최적화된다.

### 1) Fixed Out-Degree ($d$)
CAGRA 그래프의 모든 노드는 동일한 고정된 out-degree $d$를 가진다.
고정 되지 못한 $d$ 를 쓴다면 덜 중요한 거리 계산을 줄일 수 있지만, 차수가 너무 작을시 CTA(Cooperative Thread Array), 즉 스레드 
블록을 충분히 포화시킬 수 없어서 하드웨어 활용도가 낮아진다. 때문에 고정된 $d$ 값을 사용함으로 써 
GPU의 massively parallel computing 환경에서 load imbalance를 최소화하고 균일한 연산을 가능하게 하여 하드웨어 활용도를 극대화한다.

### 2) Directional
Fixed out-degree의 자연스러운 결과로 그래프는 directed graph가 된다.

### 3) No Hierarchy
HNSW와 같은 계층적 구조를 사용하지 않는다. 대신 GPU의 높은 병렬 처리 능력과 메모리 대역폭을 활용하여
무작위 샘플링을 통해 초기 노드를 효율적으로 선택한다.

## 3. CAGRA Graph Construction 및 최적화
그래프 구축은 크게 두 단계로 나뉜다.

### 1) Initial Graph Construction
- NN-Descent 알고리즘을 사용하여 k-NN graph를 초기 그래프로 구축한다. 이때 초기 out-degree $d_{init}$는 최종 CAGRA 그래프의 degree $d$의 2배 또는 3배로 설정한다.
- 각 노드의 연결된 이웃 리스트는 source node로부터의 distance 기준으로 오름차순 정렬됩니다. 이 과정은 각 노드 리스트에 대한 연산이 독립적이므로 GPU에서 효율적으로 병렬 실행 가능하다.

### 2) Graph Optimization
이 단계는 dataset이나 distance calculation이 필요 없으며, 높은 병렬성을 가진다. 

#### a. Reordering Edges
그래프의 다양성을 높이고 2-hop node counts를 증가시키기 위해 엣지(edge)의 순서를 재정렬한다.
- "detourable route" 개념을 사용한다. 노드 X에서 Y로 가는 엣지 $e_{X \rightarrow Y}$가 있을 때, 다른 노드 Z를 경유하는 경로 $(e_{X \rightarrow Z}, e_{Z \rightarrow Y})$가 $max(w_{X \rightarrow Z}, w_{Z \rightarrow Y}) < w_{X \rightarrow Y}$를 만족하면 detourable하다고 봅니다. 여기서 $w$는 distance를 의미한다.
- CAGRA는 실제 distance 대신, 초기 정렬된 이웃 리스트에서의 엣지 위치를 나타내는 "rank"를 사용하여 detourable route의 수를 근사한다 (Rank-based Reordering). 이는 distance-based reordering의 비실용적인 distance computation (O($N d_{init}^3$) 또는 대규모 distance table) 문제를 해결한다.
- 재정렬 후, 각 노드에 대해 상위 $d$개의 이웃만 남기고 pruning한다.

#### b. Reverse Edge Addition
재정렬되고 pruning된 그래프에서 모든 엣지의 방향을 반전시킨 reversed graph를 생성한다. Reversed graph의 out-degree는 고정되지 않지만 $d$로 상한이 설정됩니다. Reversed 엣지는 pruning된 그래프에서의 rank 기준으로 정렬됩니다. 이 기술은 노드 reachability를 향상시키고 strongly connected components (strong CC)의 수를 줄인다.

#### c. Merging
Pruning된 그래프와 reversed graph에서 각각 $d/2$개의 자식 노드를 선택하여 병합한다.

### 3) CAGRA Search 알고리즘
CAGRA의 검색 알고리즘은 internal top-M list (priority queue, 길이 $M \geq k$)와 candidate list (길이 $p \times d$)로 구성된 sequential memory buffer를 사용한다.

#### a. Random Sampling (Initialization)
$p \times d$개의 노드를 무작위로 샘플링하고 query까지의 distance를 계산하여 candidate list에 저장한다. Internal top-M list는 더미 값(예: FLT_MAX)으로 초기화된다.

#### b. Internal Top-M List Update
전체 버퍼에서 가장 작은 distance를 가진 상위 M개 노드를 선택하여 internal top-M list를 업데이트한다.

#### c. Candidate List Index Update (Graph Traversal)
Internal top-M list의 상위 $p$개 노드 중 이전에 parent가 아니었던 노드들의 이웃 인덱스를 가져와 candidate list에 저장한다. 이 단계에서는 distance calculation을 수행하지 않는다.

#### d. Distance Calculation
Candidate list에 있는 노드들 중 해당 query에 대해 처음으로 후보가 된 노드들에 대해서만 distance를 계산한다. 이는 불필요한 재계산을 방지한다.
이 과정은 internal top-M list의 인덱스 번호가 수렴할 때까지 반복됩니다. 최종적으로 internal top-M list의 상위 $k$개 엔트리가 ANNS의 결과로 반환된다.

### 4) GPU 최적화 기술
CAGRA는 GPU의 특성을 활용하기 위한 여러 기술을 도입한다.

#### a. Warp Splitting
32개의 스레드로 구성된 warp를 소프트웨어적으로 더 작은 "team"으로 분할한다 (예: team size 4 또는 8). 이는 128비트 메모리 로드 효율성을 높여 GPU 활용도를 극대화한다. 데이터셋 차원이 작을 때 (예: 96차원 float 데이터), 전체 warp가 아닌 팀 단위로 벡터를 로드하고 여러 번 반복하여 벡터 전체를 로드하는 것이 효율적이다.

#### b. Top-M Calculation
Internal top-M list는 이미 정렬되어 있으므로, candidate buffer를 정렬한 후 bitonic sort의 merge 과정을 통해 기존 top-M list와 병합하여 전체 연산을 줄입니다. Candidate buffer가 작을 때 (≤ 512)는 warp-level bitonic sort를 사용하여 register에서 처리하고, 클 때 (> 512)는 CTA 내에서 radix-based sort를 사용한다.

#### c. Forgettable Hash Table Management
방문한 노드 리스트 관리를 위해 open addressing hash table을 사용한다. Shared memory에 hash table을 배치하는 경우, 제한된 메모리 용량을 위해 주기적으로 리셋되는 "forgettable hash table"을 사용한다. 이는 메모리 사용량(일반적으로 ≤ 4KB)을 줄여 large-batch query에서 높은 병렬 효율성을 제공한다.

#### d. 1-bit Parented Node Management
노드가 이전에 parent 역할을 했는지 여부를 추적하기 위해 노드 인덱스 변수의 Most Significant Bit (MSB)를 플래그로 사용한다. 이는 hash table 룩업보다 빠르지만, 데이터셋 크기를 인덱스 데이터 타입 최대값의 절반으로 제한하는 단점이 있다.

### 5) 구현 선택 (Single-CTA vs. Multi-CTA)
CAGRA는 query batch size와 internal top-M size에 따라 Single-CTA 및 Multi-CTA 구현을 동적으로 선택한다.

#### a. Single-CTA Implementation
각 query를 하나의 CTA에 매핑한다. 중간에서 큰 batch size (100 이상)에 적합하며, 여러 CTA가 동시에 실행된다. Shared memory에 hash table을 두어 높은 성능을 달성한다.
대규모 데이터셋에서는 Device memory 대역폭이 병목이 될 수 있어, FP16과 같은 low-precision data type 사용을 통해 처리량을 높일 수 있다.

#### b. Multi-CTA Implementation
하나의 query를 여러 CTA에 매핑한다. 작은 batch size (1~100)에 적합하며, 단일 query에 대해서도 GPU 활용률을 높여준다.
Hash table은 여러 CTA가 공유해야 하므로 Device memory에 배치된다. Multi-CTA는 Single-CTA보다 더 많은 노드를 탐색하여 높은 recall을 달성할 수 있다.

### 6) 성능 평가
CAGRA는 HNSW (CPU SOTA), GGNN, GANNS (GPU SOTA 후보), NSSG와 비교하여 다음과 같은 결과를 보였다.

#### a. Graph Construction Time (Q-C1)
CAGRA는 HNSW보다 2.2–27배, GGNN보다 1.1–31배, GANNS보다 1.0–6.1배 빨랐다.

#### b. Graph Search Quality (Q-C2)
NSSG의 검색 구현체를 사용하여 평가했을 때, CAGRA 그래프는 NSSG 그래프와 거의 동일한 검색 성능을 보였는다.. 이는 CAGRA 그래프 자체의 품질이 우수함을 의미한다.

#### c. Large-Batch Query Throughput (Q-C3)
90%에서 95% recall 범위에서 CAGRA는 HNSW보다 33–77배, 다른 GPU 구현체보다 3.8–8.8배 빨랐다. FP16 데이터 타입을 사용하면 더 높은 처리량 이점을 얻을 수 있었다.

#### d. Single-Query Throughput (Q-C4)
95% recall에서 CAGRA는 HNSW보다 3.4–53배 빨랐다. 대규모 batch에 최적화된 GGNN 및 GANNS는 단일 query에서는 HNSW보다도 느렸다.

#### e. Large Datasets Support (Q-C5)
DEEP-1M, 10M, 100M 데이터셋에서 그래프 구축 시간은 데이터셋 크기에 비례하여 증가했다.
검색 성능은 데이터셋 크기가 커짐에 따라 recall이 약간 저하되지만, 전체적인 추세는 HNSW와 유사했으며 성능 저하는 크게 유의미하지 않았다.
이는 CAGRA가 device memory 용량 내에서 대규모 데이터셋을 효율적으로 처리할 수 있음을 보여준다.

### 7) 결론
CAGRA는 NVIDIA GPU의 고유한 아키텍처를 활용하여 graph construction 및 search operation 모두에서 탁월한 성능을 제공하는 혁신적인 ANNS 솔루션이다.
CPU 및 GPU 기반의 기존 SOTA 방법론들을 능가하는 속도와 효율성을 보여주며, large-batch와 single query 시나리오 모두에서 높은 처리량을 달성했다.

> ※ 추가 업데이트 예정이다.
{: .prompt-tip }

# 참고문헌
- Hiroyuki Ootomo, , Akira Naruse, Corey Nolet, Ray Wang, Tamas Feher, and Yong Wang. "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs." (2024).



