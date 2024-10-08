---
title: MongoDB Sharding
author: blakewoo
date: 2024-9-10 23:00:00 +0900
categories: [Database]
tags: [Computer science, Database, MongoDB]
render_with_liquid: false
---

# 샤딩(Sharding)

## 수직 확장 vs 수평 확장
일반적으로 수직확장이라고 하면 단일 서버의 CPU 성능이나 RAM, 하드디스크의 용량을 늘리는 것이고
수평 확장이라고하면 물리적인 서버를 한대 더 두는 것을 뜻한다.

이렇게만 들으면 서버를 한 대 더 사는것보다 단일 서버 성능을 늘리는게 돈이나 노력면에서 훨씬
낫지 않느냐고 말할 수 있겠지만 일반적으로 서버의 가격은 성능 대비 가격이 정비례보다 더 가파르게
오르는 선을 그리기 때문에 차라리 서버를 한 대 더 장만하는게 싸게 먹힐때도 많다.
물론 노력 부분에서는 수직 확장이 조금 더 나을 수도 있다. 동일한 서버를 한 대 더 사서 세팅하고
네트워크 구성까지 맞추느니 있는것에서 조금 더 바꾸는게 더 쉽다.

요새는 클라우드 서비스도 많이 사용하는데 이런 경우 수직 확장이라고하면 좀 더 좋은 성능의
서버 인스턴스를 빌리는 것으로도 해석이 된다. 이런 경우 물리 서버 보다는 가격에 대한 고민이 적을 순
있겠지만 이 역시 코어와 RAM의 제한을 맞이할 수 밖에 없고, 클라우드 서비스에서 해당 인스턴스보다
더 높은 성능의 인스턴스를 지원하지 않는다면 아예 수직 확장은 할 수도 없다.
그에 비해 수평확장은 동일한 성능의 인스턴스를 하나 더 빌리는 셈이니 제한이라고 할 것도 없다.

그렇다면 수평확장이 수직확장보다 무조건 좋은가?
그것도 어렵다고 볼 수 있는게, 물론 많은 부분에서 수평학장이 수직확장보다는 더 많은 이점을 가지는 것은
사실이지만 경우에 따라서 아닐 수도 있다.
이런 수평 확장은 대부분 네트워크를 통해 연결되는데 아무래도 CPU와 램, CPU와 하드디스크 간에
통신하는것보다는 월등이 느린게 네트워크이다 보니 성능에 대해서 저하가 있을 순 있다.

MongoDB에서는 수직확장보다는 수평 확장에 초점이 맞추어져 설계가 되어 있으며 수평 확장을
구현 한 것이 바로 샤딩이라는 기술이다.

## 샤딩의 개요
샤딩이란 큰 콜렉션 데이터를 여러개의 물리 서버에 나누어 적재하는 것이다.
이렇게 나누어서 적재하면 몇 가지 이점이 있다.
- 하드웨어 성능에 따른 용량 제한이 없어진다.
- 경우에 따라 병렬 처리가 가능하다.
- 특정 위치에 특정 데이터를 적재할 수 있다.
  (개인 정보 같이 데이터 위치가 중요한 경우 사용 가능)

어떨 때 이런 샤딩이 필요한지 MongoDB에서 제안하는 기준이 있는데 그 기준은 아래와 같다.

- 리소스가 최대로 활용되고 있음
- 스키마와 코드가 이미 최적화되어 있을때
- 현재 서버를 업그레이드하는 것이 비용 효율적으로 문제를 해결할 수 없을때
- 백업 복원 시간 (RTO) 목표를 충족해야 하며 이를 위해 병렬성이 필요할때

이 네 가지를 충족하면 샤딩하는것이 더욱 더 좋은 방법이라고 이야기한다.

## 샤드 클러스터의 구성요소
샤드 클러스터를 구성하기 위해서는 아래와 같은 요소가 필요하다.
### shard server
실제로 데이터가 적재되어있는 서버이다. 무조건 레플리카 셋으로 구성되어야한다.

### Config server
해당 데이터가 어디에 위치해있는지 갖고있는 서버로 config db에 해당 정보를 갖고있으며
mongos만 해당 데이터에 접근이 가능하다.

### Mongos server
몽고디비(MongoDB) 데이터베이스처럼 동작하지만 요청을 라우팅하는 서버로,
config 서버에서 데이터의 위치를 받아온 뒤 필요한 곳으로만 작업을 보낸다.
라우팅만 하기 때문에 별도의 저장장치가 필요하지 않으며 shard 서버와 config서버와
네트워크적으로 연결이 되어있어야한다.

## Shard Keys
데이터를 나누기 위해서는 무엇을 두고 나눌 것인지 기준이 필요한데 샤드 클러스터에서
이러한 기준이 되는 값을 샤드키라고 한다.
이런 샤드키를 어떻게 두느냐에 따라 성능이 천차만별로 달라지며 목적에 따라 샤드키 지정하는 방식이 달라지기도 한다.

일반적으로 샤드키를 지정하는 기준은 아래와 같다.
- 대부분의 쿼리에 포함됨
- 합리적으로 높은 Cardinality (ex 성별은 낮음, 이름은 높음)
- 이상적으로 64MB 이상의 데이터를 공유하지 않는 키
- 함께 검색하려는 데이터와 같이 공유되는 키

이런 이유로 대부분 하위 집합으로 잡는 경우가 대부분이다.
- 사용자 데이터 : 사용자별로 샤드됨, 은행 계좌 또는 게임
- 부서 데이터 : 부서 또는 지점별로 샤드됨

뚜렷한 하위 집합이 없는 경우 병렬성을 위하여 샤드키를 지정하기도하는데
분석 데이터 저장소의 경우, 샤드 키로 임의의 값을 저장하기도 한다.

좋은 샤드 키를 갖는 경우에는 아래와 같은 이점을 갖는다.
- 대부분의 작업은 단일 샤드나 소수의 샤드를 대상으로 한다.
- 개별 사용자 대부분이 단일 샤드에 접근함
- 샤드가 다운된 경우 일부 작업만 실패함
- 모든 쿼리가 모든 샤드로 이동하는 것보다 효율적이기에
  수행되는 작업이 적고 평균 대기 시간이 작다.

하지만 샤드를 특정할 수 없는 작업의 경우 전체 샤드에 요청을 보내게되는데
이런 경우는 안 좋은 샤드키를 지정했거나 적절하지 못한 쿼리일 경우이며 특정 샤드가 다운된 경우
모든 작업이 완전히 실패하게 된다.

## How Sharding works
Shard 된 데이터는 chunk 단위로 적재된다. 이 chunk에 대한 설명은 아래에 있다.

### Chunk
샤드키를 기준으로 나뉘어진 콜렉션 조각을 chunk라고 하며 아래와 같은 특징을 갖는다
- 샤드 키가 범위 내에 속하는 경우, 해당 Chunk에 포함된다.
- chunk의 각 범위의 데이터는 항상 하한값을 포함하고 상한값은 제외한다.

이러한 샤드에서 읽기/쓰기를 할 경우 특정 chunk를 갖고 있는 shard로 라우팅되는데
샤드키를 포함하지 않는 쿼리를 사용할 경우 전체 샤드에 요청을 보내게된다.
사실상 콜렉션 전체 스캔과 동일하다.
이렇게 콜렉션 전체 스캔을 사용했을 때 효율적인 경우는 큰 데이터의 병렬성을 위해서
샤드를 할때이며 이 경우도 상황에 따라 다를 수 있다.


# 참고자료
- [MongoDB 공식 문서](https://www.mongodb.com/ko-kr/docs/manual/core/document/)

