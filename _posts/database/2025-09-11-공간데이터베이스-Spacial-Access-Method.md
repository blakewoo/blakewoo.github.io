---
title: 공간 데이터 베이스 - Spatial Access Method
author: blakewoo
date: 2025-9-11 23:00:00 +0900
categories: [Database]
tags: [Computer science, Database, Spacial Database]
render_with_liquid: false
use_math: true
---

# 공간 데이터 베이스 - Spatial Access Method
이전 포스팅에서는 Spacial Query 종류가 어떤것이 있는지, 그리고 어떻게 사용하는지 예시를 알아보았다.
이번에는 어떤식으로 해당 쿼리들을 구현하는지에 대해 알아볼까 한다.

## 1. 개요
우리가 일반적으로 사용하던 SQL DB에서 일반적인 타입의 경우에는 값을 찾아오기에 매우 쉬운 편이다.   
숫자 타입이라면 크고 작음을 비교할 수 있고, 사실 문자열의 경우에도 사전 배열식으로 정렬하면 크고 작음을 비교 할 수 있기 때문에
B+ 트리의 혜택을 받아 전체 정렬(Total Ordering)이 가능하다. 즉, 어떤 값을 뽑아서 대소를 비교할 수 있기 때문에
해당 Column에 대해서 Index만 걸려있다면 $log_{m}N$ (M은 차수, N은 총 개수) 내에 찾을 수 있는 것이다.

하지만 공간 데이터의 경우 좀 애매하다.   
어떤 2차원 공간에서 좌표간의 대소를 비교하기엔 좀 어렵다. 물론 X와 Y로 나누어서 별도로 비교할 수는 있겠지만
그렇다고 X와 Y로 나누어서 별도로 INDEX를 걸어봐야 찾기 더 어려워질 뿐이다.

아래와 같은 어떤 Polygon이 있다고 해보자.

![img.png](/assets/blog/database/spacial_database/spatial_access_method/img.png)

해당 Polygon안에 어떤 점이 포함되어있는지 알고 싶다고 할때 여기서 공간 데이터 베이스가 어떻게 처리하면 좋을까?
여기서 등장하는 것이 필터와 정밀검사 두 단계이다.

## 2. 방식
### 1) Filter Step
처음에 Polygon이 추가되고 Indexing 될 때 해당 Polygon에 대해서 minimum bounding rectangle 혹은 minimum bounding box(이하 mbb)
라고 불리는 박스를 만든다. 이는 해당 Polygon이 아래와 같이 딱 들어 맞는 사각형을 말한다.

![img_1.png](/assets/blog/database/spacial_database/spatial_access_method/img_1.png)

위와 같은 mbb는 우측 상단과 좌측 하단에 있는 빨간 점과 같이 2개의 점으로 표현 가능하다.   
DB는 해당 Polygon 안에 포함된 점을 찾을 때 저 빨간 2개의 점을 이용하여 범위로 Search가 가능하다.

![img_2.png](/assets/blog/database/spacial_database/spatial_access_method/img_2.png)

위와 같은 점들이 있을 때 C는 확실히 배제하고 시작할 수 있는 것이다.   
하지만 A와 B에 대해서는 위와 같은 사각형에 포함되어있는지 확인 후에 별도로 세부 검증이 필요하다.

### 2) refinement Step
위와 같이 사각형 내에는 포함되어있으나 실제 포함되어있는지 여부를 확인하기 위해 A,B 점은 실제 기하 구조에 대해서
확인을 하는데, 이 과정에서 탈락되는 객체를 false drop이라고 부른다.


> 추가 업데이트 예정
{: .prompt-tip }

# 참고자료
- Shashi Shekhar and Sanjay Chawla, Spatial Databases: A Tour, Prentice Hall, 2003
- P. RIigaux, M. Scholl, and A. Voisard, SPATIAL DATABASES With Application to GIS, Morgan Kaufmann Publishers, 2002
