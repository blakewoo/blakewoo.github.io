---
title: 공간 데이터 베이스 - Spatial Access Method
author: blakewoo
date: 2025-9-12 23:00:00 +0900
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

해당 Polygon안에 어떤 점이 포함되어있는지 알고 싶다고 할때 여기서 공간 데이터 베이스에서는 인덱스를 어떻게 처리하면 좋을까?
여기서 등장하는 것이 필터와 정밀검사 두 단계이다.

## 2. 인덱스 방식
### 1) Space-Driven Structure
#### ※ MBB(Minimum Bounding Box)
처음에 Polygon이 추가되고 Indexing 될 때 해당 Polygon에 대해서 minimum bounding rectangle 혹은 minimum bounding box(이하 mbb)
라고 불리는 박스를 만든다. 이는 해당 Polygon이 아래와 같이 딱 들어 맞는 사각형을 말한다.

![img_1.png](/assets/blog/database/spacial_database/spatial_access_method/img_1.png)

위와 같은 mbb는 우측 상단과 좌측 하단에 있는 빨간 점과 같이 2개의 점으로 표현 가능하다.   
DB는 해당 Polygon 안에 포함된 점을 찾을 때 저 빨간 2개의 점을 이용하여 범위로 Search가 가능하다.

![img_2.png](/assets/blog/database/spacial_database/spatial_access_method/img_2.png)

위와 같은 점들이 있을 때 C는 확실히 배제하고 시작할 수 있는 것이다.

그 와중에 사각형 내에는 포함되어있으나 실제 포함되어있는지 여부를 확인하기 위해 A,B 점은 실제 기하 구조에 대해서
확인을 해야하는데, 이 과정에서 탈락되는 객체를 false drop이라고 부른다.

false drop이 적을 수록 실제 정밀 검사를 해야할 대상이 줄어드는 것이다.

꼭 인덱스를 mbb 같은 사각형으로 하지 않아도 괜찮다. 하지만 사각형 같이 근사(Approximate)한 구조체가
복잡할 수록 인덱스의 유지 비용과 검색 비용이 올라간다.

#### a. 고정 그리드(Fixed Grid)
공간을 균등한 크기의 nx × ny Cell로 나눈다. 각각의 셀은 Disk Page와 연관되어있으며
포인트 P가 어떤 Cell안에 포함되어있다면 포인트 P는 해당 Cell안에 있으며 해당 Disk Page에 포함되어있다.

각 포인트는 어떤 Cell에 포함되어있는지 알 수 있으며 해당 Cell을 해당 Page에서 불러오면 값을 읽어올 수 있다.
만약에 다수의 Cell에 대해서 연관되어있다면 대상 Cell들과 연결되어있는 모든 Page를 불러와서 처리해야한다.

만약 셀 크기를 잘 못 잡으면 특정 Cell로 많은 포인트가 몰려서 Overflow(해당 페이지에 모두 기재할 수 없는 경우)가 발생 할 수 있으며
만약 Point 자체도 셀에 균일 분포되어있는게 아니라 비균일 분포라면 성능이 저하 될 수 있다.

#### b. Grid File
Grid File은 각 축(x, y, …)에 대해 분할 경계(scale)를 관리하는 적응형 격자 디렉터리와 실제 데이터(버킷/페이지)를 분리해서 저장한다.
디렉터리의 각 셀은 물리적 데이터 페이지(버킷)를 가리키며, 여러 디렉터리 셀이 같은 데이터 페이지를 참조할 수 있어 공간을 절약한다.
삽입 시에는 Overflow가 난 셀만 국소적으로 분할하고, 필요하면 디렉터리(또는 스케일)를 확장한다.

이렇게만 들으면 이해가 잘 가지 않는다. 아래의 예시를 보자.

##### ※ 예시
입력값이 a(0,0), b(0,1), c(0,1), d(0,1), e(0,1), f(0,1), g(0,1), h(0,1), i(0,1)이고 순서대로 추가된다고 가정 해보자.   
기본적으로 x와 y에 대한 스케일은 Sx=[∞], Sy=[∞] 이다.      
Pn은 페이지 넘버이고, DIR[N,M]이 가로로 N번째 세로로 M번째 디렉토리 셀이라고 할때   
각 물리적 페이지에 총 3개씩 Point가 들어갈 수 있다고 하면 입력값이 하나씩 추가됨에 따라 디렉터리의 변화와 페이지의 변화는 아래와 같다.
디렉토리 분할이 일어나지 전까지는 그림을 통합하여두었다.

![img.png](/assets/blog/database/spacial_database/spatial_access_method/img_3.png)

- a 입력   
  P1 = a, DIR[1,1]=P1 이다. 페이지당 3개 초과가 아니기 때문에 분할은 일어나지 않는다.
  

- b 입력   
  P1 = a,b DIR[1,1]=P1 이다. 페이지당 3개 초과가 아니기 때문에 분할은 일어나지 않는다.
  

- c 입력   
  P1 = a,b,c DIR[1,1]=P1 이다. 페이지당 3개 초과가 아니기 때문에 분할은 일어나지 않는다.
  
  
![img_1.png](/assets/blog/database/spacial_database/spatial_access_method/img_4.png)

- d 입력   
  한 페이지에 3개를 초과하므로 분할이 일어난다. y축 부터 평행하게 나누게 된다면 0.5를 기준으로 나누게 되며
  때문에 Sx=[0.5,∞], Sy=[∞] 가 된다. 또한 물리 페이지 역시 분할해야되기 때문에
  P1 = a,b, P2 = c,d DIR[1,1]=P1, DIR[2,1]=P2 이다.


- e 입력   
  각 페이지별로 3개 초과가 아니므로 page에 포인트만 추가되고 그대로 유지된다.
  Sx=[0.5,∞], Sy=[∞]에 P1 = a,b,e , P2 = c,d DIR[1,1]=P1, DIR[2,1]=P2 이다.


- f 입력   
  각 페이지별로 3개 초과가 아니므로 page에 포인트만 추가되고 그대로 유지된다.
  Sx=[0.5,∞], Sy=[∞]에 P1 = a,b,e ,  P2 = c,d,f  DIR[1,1]=P1, DIR[2,1]=P2 이다.


![img_2.png](/assets/blog/database/spacial_database/spatial_access_method/img_5.png)


- g 입력   
  한 페이지에 3개를 초과하므로 분할이 일어난다. x축으로 부터 평행하게 나누게 된다면 1.5를 기준으로 나누게 되며
  Sx=[0.5,∞], Sy=[1.5,∞]에 P1 = a,b,e  , P2 = c,d , P3=f,g DIR[1,1]=P1, DIR[2,1]=P2, DIR[1,2]=P1 ,DIR[2,2]=P3 이다.


- h 입력   
  각 페이지별로 3개 초과가 아니므로 page에 포인트만 추가되고 그대로 유지된다.
  Sx=[0.5,∞], Sy=[1.5,∞]에 P1 = a,b,e  , P2 = c,d,h , P3=f,g DIR[1,1]=P1, DIR[2,1]=P2, DIR[1,2]=P1 ,DIR[2,2]=P3 이다.


![img_3.png](/assets/blog/database/spacial_database/spatial_access_method/img_6.png)

- i 입력   
  한 페이지에 3개를 초과하므로 분할이 일어난다. y축으로 부터 평행하게 나누게 된다면 1.5를 기준으로 나누게 되며
  Sx=[0.5,1.5,∞], Sy=[1.5,∞]에 P1 = a,b,e  , P2 = c,d , P3=f,g, P4=h,i, DIR[1,1]=P1, DIR[2,1]=P2, DIR[1,2]=P1 ,DIR[2,2]=P3, DIR[3,1]=P3 , DIR[3,2]=P3, 이다.

##### ※ 삽입시 세가지 케이스
위 예시를 보면서 총 세가지 케이스로 처리되는 것을 알 수 있었을 것이다. 이를 정리해보면 아래와 같다.

- A. No cell split   
  대상 페이지에 여유가 있으면 단순히 삽입 → 끝.


- B. Cell split, but no directory split   
  대상 버킷이 넘치면 그 셀을 분할(어느 축을 쪼갤지 결정)
  분할은 한 축(x 혹은 y)에서 경계 하나를 추가(스케일 배열에 값 삽입)해서 디렉터리의 일부 엔트리들이 새로 생성된 버킷을 가리키도록 함.
  디렉터리 자체(메모리 구조)의 크기는 동일하지만, 해당 행/열의 엔트리들이 새 버킷 포인터로 변경될 수 있음.


- C. Cell split and directory split   
  분할하려는 셀이 디렉터리에서 여러 엔트리의 공유 대상(즉 디렉터리의 포인트 수용량이 충분하지 않을 때)인 경우, 스케일 배열 갱신이 디렉터리의 차원을 실질적으로 '늘려야' 할 수 있음.
  이때 디렉터리를 확장 또는 재생성(일부 복제)해야 함. 일반적으로 디렉터리의 행/열을 반복 복제하거나 더 세밀한 인덱스를 만들게 됨.
  구현상 디렉터리 분할은 비용이 크므로 드물게 발생하도록 설계(버킷 분할 전략, 축 선택 규칙 등)를 조정함.


> 추가 업데이트 및 검증 예정
{: .prompt-tip }

# 참고자료
- Shashi Shekhar and Sanjay Chawla, Spatial Databases: A Tour, Prentice Hall, 2003
- P. RIigaux, M. Scholl, and A. Voisard, SPATIAL DATABASES With Application to GIS, Morgan Kaufmann Publishers, 2002
