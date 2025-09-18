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

#### c. Linear Quadtree(선형 쿼드 트리)
기본적으로 2차원 공간을 재귀적으로 4분면으로 분할하며 공간 안의 객체의 밀도를 조절하는 방식이다.   
요컨대 각 사분면에 겹치는 사각형이 페이지 용량 미만이 될 때까지 4분면으로 나누는 방식이라고 할 수 있다.   

이렇게 나눠진 사분면을 B+트리로 만들고, 이 트리의 각 리프노드들을 1차원 순서로 정렬한 뒤 차례대로 1차원 Index에(주로 B+ tree) 연결후
해당하는 디스크 공간에 넣는 방식이다.

각 리프노드는 Mbb 값과 해당 Polygon의 데이터가 포함된 Page 주소를 갖고 있으며 Mbb 값이 있기 때문에 비교적 빠른 속도로 filtering이 가능하다.

##### ① 인덱싱 방법
- Z-order

순서를 Z 모양으로 구성한다고 해서 Z-order이다.   
어떤 2차원 평면이 있다고 할 때 4분면으로 나누면 아래와 같은 그림으로 순서를 잡는다.

![img.png](/assets/blog/database/spacial_database/spatial_access_method/img_12.png)

만약 4개가 아닌 각각의 사분면에 한번 더 잘릴 경우 아래와 같다.

![img_1.png](/assets/blog/database/spacial_database/spatial_access_method/img_13.png)

위와 같이 계속해서 Z 형태로 연속해서 순서를 부여한다. 이때 순서에 대한 문자는 사전식 정렬(Lexicographical Order)을 따르며
이 순서를 기준으로 B+ tree에 Mapping한다.

이러한 Z-order는 키 계산이 Hilbert curve 보단 단순해서 연산 리소스를 크게 잡아먹진 않는다.   
하지만 2차원 평면 상에서 데이터가 가까울 경우에 1차원 인덱스까지 가깝진 않은데 이는 검색시에   
쓸데 없는 데이터까지 갖고오는 문제를 초래할 수 있다.

- Hilbert curve

앞서 말했듯이 Z-order의 경우에는 2차원에서 원래 데이터가 가깝다고 1차원 인덱스까지 가깝진 않다.  
하지만 지금 이야기하는 Hilbert curve의 경우에는 Z-order 보다는 좀 더 가까워서(지역성이 좋다고 말한다)   
효율적이다.

기본적으로 2차원 평면이 있을 때 이를 4분할한 뒤 순서를 아래와 같이 곡선으로 부여한다.

![img_2.png](/assets/blog/database/spacial_database/spatial_access_method/img_14.png)

만약 각 사분면의 데이터가 용량을 초과하여 추가적으로 분할해야할 경우 아래와 같이 순서를 부여한다.

![img_3.png](/assets/blog/database/spacial_database/spatial_access_method/img_15.png)

이는 기본적인 U 자형 곡선을 동일한 규칙으로 패턴을 만들어 프랙탈과 같이 배치하는 형태이다.

이러한 Hilbert curve는 2 차원 평면상에서 각 데이터의 근접성을 Z-order에 비해 비교적 잘 보존한다
때문에 범위 질의(특히 작은 지역)의 I/O 감소할 수 있다. 하지만 키 계산이 Z-order보다 복잡(회전/반전 상태 관리 필요)하고
2 차원 공간의 모든 이웃 관계를 완벽히 보존할 수는 없다.

##### ② 예시
페이지당 4개의 Polygon 정보를 담을 수 있다고 할 때 아래의 그림을 보자. 그림에서 Order는 Z-order로 표현했다.

![img.png](/assets/blog/database/spacial_database/spatial_access_method/img_7.png)

위와 같이 2차원 공간 R에 다음과 같이 다각형의 MBB가 있다. 먼저, R을 4분면으로 나눈다.

![img_1.png](/assets/blog/database/spacial_database/spatial_access_method/img_8.png)

위를 트리로 나타내면 아래와 같다.

![img_2.png](/assets/blog/database/spacial_database/spatial_access_method/img_9.png)

각 리프노드는 페이지이다. polygon이 다른 분면에 걸쳐있다면 각 분면에 복제하여 같이 저장한다.
페이지당 4개의 Polygon 정보를 담을 수 있는데, 리프노드 0을 보면 4개가 초과되어있다.   
따라서 리프노드 0을 분할하기 위해 공간을 다시 4분면으로 분할 한다.

![img_3.png](/assets/blog/database/spacial_database/spatial_access_method/img_10.png)

위와 같이 분할된 값을 다시 트리로 그리면 아래와 같다.

![img_4.png](/assets/blog/database/spacial_database/spatial_access_method/img_11.png)

보면 알겠지만 각 분면에 겹치는 mbb가 많을 수록 데이터의 중복이 많은 것을 알 수 있다.    
아래는 15와 16가 추가 되는 경우 어떻게 분할되는지를 설명하는 그림이다.

![img_4.png](/assets/blog/database/spacial_database/spatial_access_method/img_16.png)

보면 알겠지만 사분면 1과 15가 추가되었고 0,1에 걸쳐 16이 추가된 걸 볼 수 있다.   
이 경우 사분면 1의 용량이 초과되었음로 4분할 하여 10,11,12,13으로 만든다.   
16의 경우 사분면의 용량이 초과하지 않았으므로 분할없이 추가된다.   

이를 트리로 나타내면 아래와 같다.

![img_5.png](/assets/blog/database/spacial_database/spatial_access_method/img_17.png)


##### ③ 단점
위 방식의 단점을 정리하면 아래와 같다.

- 소수의 자식 노드는 페이지의 작은 부분만 차지하는 4개로 고정된다.
- 쿼드트리 쿼리 시간은 트리 깊이와 관련이 있으며, 깊이가 클 수 있다.
- 페이지에 들어간 polygon의 중복이 매우 많다.

> 추가 업데이트 및 검증 예정
{: .prompt-tip }

# 참고자료
- Shashi Shekhar and Sanjay Chawla, Spatial Databases: A Tour, Prentice Hall, 2003
- P. RIigaux, M. Scholl, and A. Voisard, SPATIAL DATABASES With Application to GIS, Morgan Kaufmann Publishers, 2002
