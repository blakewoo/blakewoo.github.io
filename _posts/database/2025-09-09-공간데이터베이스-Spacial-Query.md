---
title: 공간 데이터 베이스 - 공간 데이터 쿼리
author: blakewoo
date: 2025-9-9 18:00:00 +0900
categories: [Database]
tags: [Computer science, Database, Spacial Database]
render_with_liquid: false
use_math: true
---

# 공간 데이터 베이스 - 공간 데이터 쿼리

> ※ 이번 포스팅을 이해하기 위해서는 SQL에 대한 기본적인 이해가 필요하다. 기본적인 SQL에 대한 내용은 같은 카테고리내에 잘 정리되어있으니
참고하면 좋다.
{: .prompt-tip }

## 1. 개요
SQL도 버전이 있다. 1986년에 나온 SQL1, 1992년에 나온 SQL2, 1999년에 나온 SQL3가 바로 이런 버전이다.   
각 버전마다 지원하는 범위가 다른데, 이번에 다룰 공간 데이터 베이스에 대한 쿼리를 쓰기 위해서는 SQL3를 사용해야한다.   
왜냐하면 SQL1,2는 재귀적인 쿼리에 적합하지 않기 때문이다. 

공간 데이터베이스에서 사용하는 쿼리는 기본적인 RDBMS에서 사용하는 SQL + $\alpha$ 의 형태로 되어있다.    
기본 데이터형식과 기본 연산자에서 공간 데이터 타입 및 연산자를 더한 형태라는 뜻이다.


## 2. OGIS(Open Geodata Interchange Standard) Spatial Data Model
지리공간(geospatial) 데이터의 상호운용성과 교환을 위해 공개·표준화된 규격들의 집합을 말한다.
즉 서로 다른 GIS 소프트웨어·서비스 간에 좌표·속성·Geometry를 일관성 있게 주고받도록 규정한 인터페이스·포맷·프로토콜인데
많은 DB 제공사에서 제공한다.(Oracle, IBM 등)

### 1) 지원 타입
기본적으로 4가지 타입을 지원한다.

- Point : 0차원, 특정 위치를 말한다.
- Curve : 1차원, 연속된 점으로 이루어진 선으로 LineString과 LineRing 등의 구체적인 타입이 있으며, 시작 끝점이나 close여부, 자기 교차 여부등으로 판단할 수 있다.
- Surface : 2차원, 다각형이다. 내부, 경계, 구멍등의 개념을 포함한다.
- GeometryCollection/Multi : 서로 다른 타입을 묶는 컨테이너로 교차 결과에 따라 여러타입이 동시에 나올 수 있다.

### 2) 지원 연산자
아래의 3가지 카테고리의 연산자를 지원한다.

#### a. 모든 Geometry 타입에 적용되는 기본 연산(Apply to all geometry types)
- SpatialReference : 기하(Geometry)는 어떤 좌표계(SRS/SRID)에 정의되는지
- Envelope : 기하의 최소 바운딩 박스(사각형)를 반환
- Export : 기하를 다른 형식(Well-Known Text, Well-Known Binary, GML 등)으로 직렬화/내보내기
- IsSimple : 기하가 ‘단순한가’(자기-교차 등 위반이 없는가)를 검사
- Boundary : 기하의 위상학적 경계(예: 폴리곤의 외곽선과 구멍들의 링)를 반환

#### b. 위상적 술어(Predicates for Topological relationships)
- Equal : 두 기하가 동일한 점 집합을 갖는가
- Disjoint : 공통 점이 전혀 없는가
- Intersect : 적어도 하나의 점을 공유하는가, disjoint의 정반대
- Touch : 경계는 만나지만 내부는 겹치지 않는가
- Cross : 서로 다른 차원의 부분이 교차하는 경우
- Within : A가 B안에 있는가?
- Contains : B가 A안에 있는가?

#### c. 공간 데이터 분석(Spatial Data Analysis)
- Distance : 두 기하 사이의 최소 거리(숫자)
- Buffer : 기하로부터 일정 반경(거리)만큼 확장한 영역(주로 폴리곤 결과)
- Union : 두 기하를 합해 하나의 기하(또는 멀티기하)를 생성
- Intersection : 두 기하의 겹치는 부분만 반환
- ConvexHull : 기하를 완전히 감싸는 최소 볼록 폴리곤을 반환
- SymDiff : 두 기하의 공통부분을 제외한 합집합 부분

## 3. 실사용 예시
아래와 같은 스키마가 있다고 해보자.

```sql
CREATE TABLE Country(
    Name varchar(30),
    Cont varchar(30),
    Pop Integer,
    GDP Number,
    Shape Polygon);
)
```

```sql
CREATE TABLE River(
    Name varchar(30),
    Origin varchar(30),
    Length Number,
    Shape LineString);
)
```

```sql
CREATE TABLE City(
    Name varchar(30),
    Country varchar(30),
    Capital varchar(1),
    Pop Integer,
    Shape Polygon);
)
```

### 1) Area
면적을 구하는 함수로, C.Shape를 인자로 주는 형태이다. 이는 Polygon type이라 사용가능하다.
```sql
SELECT  C.Name, C.Pop,  Area(C.Shape)  AS  "Area"
FROM  Country  C
```

### 2) Distance
모든 국가의 이름과 GDP 그리고 국가의 수도와 적도까지의 거리를 출력하는 쿼리이다. 

```sql
SELECT  Co.GDP,  Distance(Point(Ci.Shape.x, 0), Ci.Shape)  AS  "Distance"
FROM  Country  Co, City  Ci WHERE  Co.Name = Ci.Country AND  Ci.Capital ='Y'
```

### 3) Touch
미국(USA)의 이웃 국가 이름을 출력하는 쿼리로, 접해있는지 확인 가능한 Touch라는 함수를 써서 구현되었다.

```sql
SELECT  C1.Name  AS  "Neighbors of  USA"
FROM  Country  C1, Country  C2
WHERE  Touch(C1.Shape,C2.Shape) = 1 
AND  C2.Name ='USA'
```

### 4) Cross
나라를 가로지르는 강이 있다면 해당 강의 이름과 나라 이름을 같이 출력하는 쿼리로, 서로 교차하는 것을 검출 할 수 있는 Cross 함수를 사용했다.

```sql
SELECT  R.Name, C.Name
FROM  River R, Country C
WHERE  Cross(R.Shape,C.Shape) = 1
```

### 5) Buffer와 Overlap
세인트 로렌스 강은 300km 이내에 있는 도시에 물을 공급할 수 있다고 할때, 세인트 로렌스 강에서 공급되는 물을 사용할 수 있는 도시를 구하려면
아래와 같은 쿼리를 사용하면 된다.

```sql
SELECT  Ci.Name
FROM  City Ci, River R
WHERE  Overlap(Ci.Shape, Buffer(R.Shape,300)) = 1 
AND  R.Name ='St.Lawrence'
```

여기서 Buffer는 해당 Shape의 입력 받은 반경을 포함한 Polygon을 반환하고 Overlap은 해당 Polygon과 City들이 겹치는지 확인하는 함수이다.



# 참고자료
- Shashi Shekhar and Sanjay Chawla, Spatial Databases: A Tour, Prentice Hall, 2003
- P. RIigaux, M. Scholl, and A. Voisard, SPATIAL DATABASES With Application to GIS, Morgan Kaufmann Publishers, 2002
- [INTERNATIONALSTANDARD-ISO19107 Geographic information — Spatialschema](https://cdn.standards.iteh.ai/samples/66175/92416c4eb8954655905aa1d18f244afc/ISO-19107-2019.pdf)
- OpenGIS® Implementation Standard for Geographic information - Simple feature access - Part 1: Common architecture -  Open Geospatial Consortium
