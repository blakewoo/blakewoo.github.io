---
title: 공간 데이터 베이스 - 데이터 쿼리
author: blakewoo
date: 2025-9-9 18:00:00 +0900
categories: [Database]
tags: [Computer science, Database, Spacial Database]
render_with_liquid: false
use_math: true
---

# 공간 데이터 베이스 - 데이터 쿼리

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

- Point
- Curve
- Surface 
- GeometryCollection

### 2) 지원 연산자
아래의 3가지 카테고리의 연산자를 지원한다.

#### a. Apply to all geometry types
- SpatialReference, Envelope, Export,IsSimple, Boundary Predicates for Topological relationships

#### b. Predicates for Topological relationships
- Equal, Disjoint, Intersect, Touch, Cross, Within, Contains Spatial Data Analysis

#### c. Spatial Data Analysis
- Distance,Buffer,Union, Intersection, ConvexHull, SymDiff

> ※ 포스팅 내용 추가 업데이트 및 검증 예정
{: .prompt-tip }


# 참고자료
- Shashi Shekhar and Sanjay Chawla, Spatial Databases: A Tour, Prentice Hall, 2003
- P. RIigaux, M. Scholl, and A. Voisard, SPATIAL DATABASES With Application to GIS, Morgan Kaufmann Publishers, 2002
- [INTERNATIONALSTANDARD-ISO19107 Geographic information — Spatialschema](https://cdn.standards.iteh.ai/samples/66175/92416c4eb8954655905aa1d18f244afc/ISO-19107-2019.pdf)
