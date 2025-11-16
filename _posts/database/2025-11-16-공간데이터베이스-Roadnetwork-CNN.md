---
title: 공간 데이터 베이스 - Road network CNN
author: blakewoo
date: 2025-11-16 23:00:00 +0900
categories: [Database]
tags: [Computer science, Database, Spacial Database,Road network,CNN]
render_with_liquid: false
use_math: true
---

# Road network CNN
## 1. 개요
이전에 포스팅했던 연속 최근접 이웃은 쿼리 선분(query segment)을 따라 이동하면서 각 구간별로 가장 가까운 데이터 포인트를 찾는 문제였다.      
하지만 실제 우리가 사는 세상에서는 별도의 도로나 길이 존재하며 이 역시 Road network에 적용된 버전이 있다.   
이번에 포스팅할 내용은 도로 네트워크 환경에서 효율적으로 연속 최근접 이웃 쿼리를 처리하는 알고리즘에 관한 내용이다.

## 2. UNICONS (UNIque CONtinuous Search algorithms) for NN queries
UNICONS는 NN 쿼리를 위한 알고리즘으로, Dijkstra 알고리즘에 사전 계산된 최근접 이웃 리스트를 통합한 방식이다.
세 개 이상의 엣지가 만나는 노드를 교차점(intersection point)이라 하며, 사전 계산된 NN 정보를 유지하는 교차점을 응축점(condensing point)이라고 한다.

### 1) 두 개의 기본 아이디어0
- 경로를 따라 연속 검색을 수행하기 위해 경로상의 객체들을 검색하고 각 노드에서 정적 쿼리를 실행하는 것으로 충분하다는 것이다   
- 두 번째 아이디어는 이동하는 쿼리 포인트와 정적 객체 사이의 네트워크 거리 변화를 구간별 선형 방정식으로 표현할 수 있다는 것이다.

### 2) 알고리즘 절차
알고리즘은 분할 정복 방식을 기반으로 다음 세 단계로 진행된다
- 1단계: 교차점을 기준으로 쿼리 경로를 부분 경로(subpath)로 분할한다.
- 2단계: 각 부분 경로에 대해 유효 구간(valid interval)을 결정합니다. 이는 다시 4개의 하위 단계로 구성된다:
  - 2.1단계: 부분 경로상의 객체들을 검색
  - 2.2단계: 부분 경로의 시작점과 끝점에서 NN 쿼리 실행
  - 2.3단계: 커버 관계를 사용하여 중복 튜플 제거
  - 2.4단계: 부분 경로를 유효 구간으로 분할
- 3단계: 인접한 부분 경로들의 유효 구간을 병합한다

> ※ 업데이트 및 추가 검증 예정
{: .prompt-tip }

# 참고자료
- Shashi Shekhar and Sanjay Chawla, Spatial Databases: A Tour, Prentice Hall, 2003
- P. RIigaux, M. Scholl, and A. Voisard, SPATIAL DATABASES With Application to GIS, Morgan Kaufmann Publishers, 2002
