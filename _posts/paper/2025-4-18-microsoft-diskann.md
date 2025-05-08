---
title: Microsoft DiskANN 논문 해석
author: blakewoo
date: 2025-5-4 23:30:00 +0900
categories: [Paper]
tags: [Paper, Microsoft, VectorDB, DiskANN] 
render_with_liquid: false
use_math: true
---

# DiskANN
## 1. 개요 
DiskANN은 그래프 기반의 대규모 근사 최근접 이웃 탐색 시 스템으로, 64 GB 메모리와 저렴한 SSD만으로
수십억 개의 벡터 를 색인 및 검색할 수 있도록 설계되었다. 
이는 전통적인 메모리 기반 근사 최근접 이웃 탐색방식이 요구하는 대용량의 RAM 용량 한계를 피하면서도,
높은 재현율과 낮은 탐색 지연을 균형있게 제공한다.

## 2. Vamana 그래프 구성 알고리즘
DiskANN의 가장 핵심은 Index를 구성하는데 핵심 알고리즘인 Vamana 그래프 구성 알고리즘이다.
Vamana 그래프 구성 알고리즘은 DiskANN의 색인을 구성하는 주요 알고리즘이다. 먼저 전체 데이터 세트 N개의 벡터에서 무작위 edge를 만든다.
여기서 기준점을 선택하고 선택된 기준 점을 기준으로 하이퍼파라미터 α값에 따라 두 번의 가지치기를 한다.
어떤 점 p의 이웃을 선택한다고 했을 때 이웃이 아닌 점들 의 집합 V라고 하자.
V에서 가장 가까운 이웃 p*를 선택한 뒤 V 에서 다른 후보점 p’을 선택할지를 이 α값으로 정하게 된다.
α값은 두 개의 벡터가 주어졌을 때 거리를 구하는 함수가 D라면 아래의 식으로 나타낼 수 있다.

$$ \alpha \times D(p*,p') \le D(p,p') $$

만약 α가 1이라면 p와 p’의 거리가 p*와 p’의 거리의 1배수보다 크거나 같아야 계속 이웃 후보로 고려한다는 뜻이다. 
만약 1배 수 보다 작다면 p’은 후보에서 제외해버리고 edge를 끊어버린다. 첫 번째 가지치기는 α값을 1로 두고 edge를 끊으며 두 번째 가지 치기에는
α를 1보다 큰 값으로 두고 이를 만족하는 edge는 다시 이어준다.

## 3. Beam Search
탐색은 그래프의 중심점인 centroid와 가장 가까운 벡터인 medoid에서 시작되며,
쿼리와의 거리가 가까운 이웃 벡터들을 Beam Width만큼 선택하여 탐색 영역을 점차 확장한다.
(여기서 Beam Width는 DiskANN을 구동하는데 인자값 W이다)

이때 거리 계산을 위해 실제 벡터 값이 필요한데, DRAM 또는 SSD의 캐시 영역에서 해당 벡터를 읽어오게 된다.
여기서는 Search List가 중요하다. 이는 DiskANN을 구동할때 인자값인 L에 해당하는데, 실질적으로 몇 개까지의 노드를 탐색할지
정하는 파라미터이다. 내부적으로 확장노드를 기록하는 집합과, 후보군을 담는 큐가 있다.
확장노드 노드를 기록(탐색이 완료된)하는 집합의 크기가 Search 실행시 입력한 Search list 크기만큼 도달하게되면 탐색이 종료된다.

이 과정에서 탐색 완료된 집합의 크기를 쿼리와 거리 기준 오름차순으로 정렬한 후 K개 만큼 반환하게 된다.

## 4. 성능에 영향을 미치는 요인
Search시 응답 시간에 영향을 주는 인자는 Beam Width에 영향을 주는 W값과 메모리에 얼마나 캐시할지 정하는 Cache 값과 Search list값이고
recall rate(재현율)에 영향을 주는 것은 Search list의 값이다.   
Search list의 크기가 늘어난다면 Recall rate는 증가하지만, 응답시간은 줄어든다.

# 참고문헌
- Jayaram Subramanya, Suhas, Fnu, Devvrit, Harsha Vardhan, Simhadri, Ravishankar, Krishnawamy, and Rohan, Kadekodi. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." . In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2019.
- 2025년 전반기 한국정보과학회 데이터베이스 투고 논문 - 우지훈, 정보돈, 정연우

