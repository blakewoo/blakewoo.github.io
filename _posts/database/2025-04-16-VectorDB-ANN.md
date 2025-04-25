---
title: Vector DB - ANN(approximate nearest neighbor)
author: blakewoo
date: 2025-4-18 18:00:00 +0900
categories: [Database]
tags: [Database, DBMS ,VectorDB]
render_with_liquid: false
use_math: true
---

# Vector DB - ANN(approximate nearest neighbor)
## 1. 개요
가령 X,Y 값 두 개로 이루어진 점의 집합인 데이터가 있다고 생각해보자.   
여기서 새로운 점이 추가되었을 때 이 점과 가장 가까운 K개의 데이터를 찾을 때 가장 좋은 방법은 무엇일까?   

여기서 좋은 방법이라는 것은 사실 굉장히 모호한 표현이고 아래와 같이 나눌 수 있다.

1) 제일 정확한 방법   
2) 제일 빠른 방법

제일 정확한 방법은 너무 명료하다. 어떤 것보다 가장 가깝다라는 것은 기본적으로 전체를 확인했을때가 가정되어있는 경우이다.   
수학적 수식이나 논리적 귀결에 의해 나머지 경우를 배제할 수 있는 경우가 아니라면(이것도 심지어 전처리가 되어있어야하는 경우가 많다,
가령 정렬되어있다던가) 대부분 전체를 확인해야한다.

전체 데이터 갯수가 N개라고 할 때 시간복잡도는 O(N)이 된다.   
왜냐하면 모든 점을 순회하며 각 점에 대해서 추가될 점에 대한 거리를 구해야하고, 그 중에 가장 가까운 것 K개 만큼을 갖고 있어야하기 때문이다.

그렇다면 제일 빠른 방법은 무엇일까?    
사실 가장 빠르기만 해선 안된다. 막말로 가장 빠른 것만 생각한다면 아무점이나 찍어서 K개만큼 반환 해줄 수도 있다.   
(이게 뭐야 할수도 있지만, 일단 빠르긴하다 ;;)   
빠른 것과 동시에 어느정도 정확성까지 보장되어야한다. 여기서 시작된 방식이 바로 유사 근접 이웃(ANN)을 찾는 방법이다.   

## 2. 종류
당연하겠지만 매우 많은 종류가 나왔다. 아래는 그 종류에 대한 설명이다.

### 1. 해시 기반
#### 1) LSH(Locality Sensitive Hashing)
유사 입력 항목을 높은 확률로 동일한 버킷에 해싱하는 퍼지 해싱기술이다.   
해시 충돌을 최소화하지 않고 최대화 한다는 점에서 기존 해싱과는 다르며 해싱 자체를 차원을 줄이는 방식으로 사용하는 방식이다.

#### 2) L2H(Learning to Hash)
데이터 분포를 이용하여 해시함수를 학습하는 방법이다.    
딥러닝 방식을 이용해서 하고 있는 데이터를 통해 모델을 만들어서 학습한다.
데이터의 분포(distribution) 혹은 클래스 레이블(label) 정보를 목적 함수에 적용하기 때문에 가능하다.

### 2. Partition-based Methods
전체 고차원 공간을 여러개의 비중첩 영역으로 나누는 방식이다.     
쿼리 벡터가 어떤 영역 A에 위치해있다면 최 근접 이웃들은 A나 A의 근처 영역에 존재한다는 것을 가정하고 탐색을 한다.   
기본적으로 이 방식은 Index를 생성할 때 Tree나 Forest 구조로 표현되며 아래의 세 유형으로 나눌 수 있다.

- 피벗 기반 분할   
  무작위로 선택된 지점(피벗)으로 부터 거리를 기준으로 데이터를 분할 한다.
  ex) VP-Tree, Ball Tree

- 초 평면 분할   
  무작위 방향의 초 평면이나 축 정렬된 초평면을 이용해 공간을 재귀적으로 나눈다.
  ex) Annoy, Rnadom-Projection Tree, Randomized KD-trees

- 컴팩트 분할   
  데이터를 클러스터로 나누거나 근사적인 보르노이 분할을 만들어 지역성을 활용한다.


### 3. 그래프 기반
가장 핫하다. 이는 동일한 재현율(recall rate : 0.8) 기준으로 이 방식이 가장 성능이 좋기 때문이다.   
여기서 크게 두 가지로 나뉜다.   

#### 1) HNSW (Hierarchical navigable small world)
기본적으로 메모리 기반 인덱싱이다.
이름에서 알수있듯이 계층적인 탐색 가능한 작은 세계이다.
원래는 NSW(navigable small world) 에서 시작한 알고리즘이다.

원래 KNN, 그러니까 전체에 대해서 계산하는건 노드가 추가될때마다 전체에 대해서 계산해야하므로 매우 비효율적인데
이 NSW(Navigable Small World)는 임의로 선택된 노드에서 지정된 수준까지만 탐색을 수행하는 방식이라
완전히 동일하진 않지만 거의 근사한 값을 빠르게 뽑아낼 수 있다.

여기서 계층적인 부분은 Skip List에서 파생된 것인데 계층이 있고 상위 계층은 적은 수의 노드를 포함하고 있지만
하위 계층일 수록 더 많은 노드를 포함하며, 제일 아래층은 모든 노드를 포함한 형태의 구조이다.
전체 구조를 가로지르며 데이터를 빠르게 탐색할 수 있는 형태인데, 여기에 NSW를 결합한 형태가 바로 HNSW인 것이다.


#### 2) DiskANN
메모리 기반 인덱싱의 경우 Data가 많아지면 메모리가 너무 많이 필요하다!   
메모리는 조립 컴퓨터를 맞춰본 사람이라면 알겠지만 매우 비싼 자원이다.   
그렇다면 메모리를 좀 덜 쓰는 방법이 없을까를 생각하다가 나온 방법이라고 할 수 있다.   
이 방식 자체는 Microsoft의 연구팀에서 발표했으며 관련 github repo까지 있다.   
DiskAnn의 방식을 간략하게 설명하자면 이 방식은 SSD의 사용을 상정하고 만든 알고리즘이다. 

벡터 데이터를 가지고 인덱스를 빌드한다. 모든 벡터 데이터간에 랜덤하게 간선을 연결해서 랜덤 그래프를 만든다.
이후 가치지기를 통해 간선 수를 줄인다. 이때 간선수를 줄이는 것을 위해 $$ \alpha $$ 값을 사용한다.
여기서 $ \alpha $ 값이란 아래와 같은 값이다.

$$ Distance(v1,v3) > \alpha \times \times (Distance(v1,v2) + Distance(v1,v3)) $$

$ \alpha $ 값이 1.2 일 경우 위 식에서 1.2까지해도 수식이 참이라면 끊지 않고 유지한다는 뜻이다.
이후 가지치기 한 것에서 추가적으로 가지를 더 연결하는데 이때는 $ \alpha = 1.2$ 값을 사용한다.
이 말인 즉슨 긴 간선을 유지하기 위함이다.

이후 완성된 그래프 중에 일부와 양자화를 통해 압축된 임베딩 벡터는 메모리에, 나머지 그래프 인덱스와 전체 임베딩 벡터의 경우
SSD에 둔다. 

이후에 특정 쿼리가 들어왔을 때 탐색 시작점(Entry point)에서 PQ 거리로 가장 가까운 후보를 찾아서 이를 탐색큐에 넣고
메모리에 해당 후보가 있다면 그걸로 체크, 없다면 비동기적으로 SSD에 요청을 보내서 값들을 읽어온다.
해당 노드에 대한 벡터값들을 읽어온 뒤 쿼리와 거리 계산을 하고 가장 높은 값들을 반환한다.

> ※ 추가 업데이트 및 검증 예정이고, 아직 완벽하지 않으니 참고만 바란다.
{: .prompt-tip }


# 참고자료
- Approximate Nearest Neighbeor Search on High Dimensional Data - Experiments, Analyses, and Improvement, TKDE'19
- [ritvikmath - Approximate Nearest Neighbors : Data Science Concepts](https://youtu.be/DRbjpuqOsjk)
- [DataMListic - Vector Database Search - Hierarchical Navigable Small Worlds (HNSW) Explained](https://youtu.be/77QH0Y2PYKg)
- [위키백과 - Locality Sensitive Hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
- [제리의 AI & OPS - 검색증강생성(RAG) - 그래프 기반 벡터 인덱스 HNSW(Hierarchical Navigable Small World)](https://jerry-ai.com/30)
