---
title: MongoDB ReplicaSet
author: blakewoo
date: 2024-9-9 21:20:00 +0900
categories: [Database]
tags: [Computer science, Database, MongoDB]
render_with_liquid: false
---

# MongoDB ReplicaSet

## Replica set
물리적인 복사본을 가지고 DB를 운용하는 것이다.
어째서 이렇게 복사본을 운용하는가 하면 총 세가지 이유가 있다.
### 높은 가용성
다수의 복사본이 있기 때문에 서버 한 개가 셧다운 되어도 다른 서버가 읽거나 쓰는 기능을
대신할 수 있기 때문에 가용성에는 문제가 없다.
단, 데이터 정합성 때문에 주(Primary node)에서만 쓰기가 가능하며 이러한 Primary Node가
셧다운 될 경우 Secondary Node가 Primary로 승격되어 쓰기 작업을 하게된다.
어떤 Secondary Node가 선택되는지는 후에 기술하겠다.

### 낮은 읽기 지연
다수의 복사본이 있으며 이러한 복사본이 물리적인 거리가 좀 떨어져있을 경우
설정에 따라 가장 가까운 복사본(지연시간이 짧은 Node)에서 읽어올 수 있고
이러한 경우 응답 속도가 빠르다.

### 다른 엑세스 패턴에 대한 대응
모든 DB가 그렇듯 캐싱이 있다. 사용자의 90%는 최근 변경 내역에 대해서만 쓰거나 읽기가 되고
10%는 DB 전체에 대해서 엑세스하는데 이렇게 전체에 대해서 엑세스 하는 경우는 전체 데이터에 대한
분석이 필요할때 이런 형태로 이루어진다. 문제는 이러한 전체 데이터 엑세스는 서버의 캐싱에 영향을 줘서
성능을 저하시키는데 Replica Set의 노드 한 개를 분석용으로 두고 해당 분석용에서 갖고올경우
전체 캐싱에는 영향을 주지 않아 성능에 영향을 끼치지 않는다.

## 레플리카 셋 노드의 종류
노드는 크게 네 종류가 있다.
Primary : 읽기 쓰기가 가능한 주 노드이다.
Secondary : 읽기만 가능한 노드 이다. Primary가 셧다운 되었을 때 Primary Node로 승격 가능하다.
Arbiter : Secondary가 셧다운 되었을 때 투표 권만 행사 할 수 있는 노드이다. 읽기나 쓰기를 지원하지 않는다.
Non-voting Members : 읽기만 가능한 노드이지만 승격과 투표가 불가능한 노드이다.

여기서 Non-voiting Members의 경우 사실 필수라고 보긴 어렵고, arbiter는 Secondary가 대신 할 수 있다.
aribter는 일반 상용 서비스에서 권장하지 않는다.

## 레플리카 셋의 구조

### 레플리카셋 구성 형태
전체 노드 개수는 최대 50개의 제한을 두고 있으나 기본적으로 투표를 할 수 있는 노드는 3, 5, 7이어야한다.
이는 MongoDB 커뮤니티 버전과 엔터프라이즈, Atlas 모두 동일하다.
괜히 홀수 인게 아닌데, 이는 N:N 형태가 되어 선정되지 않는 형태를 막기 위해서이다.

가장 많이 쓰는 형태 두 개만 소개하겠다.

- PSS : Primary, Secondary, Secondary로 이루어진 세트이다. Primary가 죽었을때 Secondary 두 개 중
  한 개가 Primary가 된다.

- PSA : Primary, Secondary, Arbiter로 이루어진 세트이다. Primary가 죽었을때 Secondary가 Primary가 된다.
  데이터 복제가 실질적으로 두 개만 되는 것으로 가용성 면에서는 PSS 보다 많이 떨어진다.

여기서 S가 더 붙거나 A가 더 붙거나하는 방식이며 P를 제외한 노드는 짝수개가 되어야한다.

### 레플리카셋 데이터 복제 절차
- 애플리케이션은 모든 변경 사항을 주 서버(Primary)에 기록한다.
- 주 서버는 시간 T에 변경 사항을 적용하고, 그것을 자신의 작업 로그(Oplog)에 기록한다.
- 복제 서버(Secondaries)는 이 작업 로그를 관찰하고, 시간 T까지의 변경 사항을 읽어온다.
- 복제 서버는 시간 T까지의 새로운 변경 사항을 자신에게 적용한다.
- 복제 서버는 자신의 작업 로그에 이를 기록한다.
- 복제 서버는 시간 T 이후의 정보를 요청한다.
- 주 서버는 각 복제 서버에 대해 가장 최근에 확인된 시간 T를 알고 있다.

### oplog
문서를 변경할 때마다 한 개의 도큐먼트에 기록되며, 컬렉션 삭제 또는 인덱스 생성과 같은 변경 내역이 있을시 기록된다.
local이라는 Database에 있으며 oplog.rs라는 capped colletion에 읽기 전용으로 들어간다.
크기는 처음 생성될때 남은 디스크 크기의 5% 정도가 기본으로 잡히며 이는 설정을 통해서 사이즈 조정이 가능하다.
(단, capped collection이지만 Mongodb 4.0부터는 처리가 되지 않은 oplog는 삭제되지 않는다)
oplog를 담는 collection이 너무 크거나 작으면 성능상에 문제가 생길 수 있으며 RAM의 크기와 맞추는 것이 좋다.
위에 복제절차에서 언급되었듯이 이러한 oplog를 통해서 Secondary Node가 동기화를 한다.

이러한 oplog는 몇 가지 특징이 있다.

1. 다수의 document 변경이라도 한 개씩 기록된다.   
   ex)
   ```javascript
    db.cols.deleteMany({type:"test"})
   ```
   ```
   { "ts" : Timestamp(1407159845, 5), "h" : NumberLong("-704612487691926908"), 
    "v": 2, "op" : "d", "ns" : "d.cols", "b" : true, "o" : { "_id" : 12 } }
    { "ts" : Timestamp(1407159845, 1), "h" : NumberLong("6014126345225019794"),
    "v": 2, "op" : "d", "ns" : "d.cols", "b" : true, "o" : { "_id" : 33 } }
    { "ts" : Timestamp(1407159845, 4), "h" : NumberLong("8178791764238465439"),
    "v": 2, "op" : "d", "ns" : "d.cols", "b" : true, "o" : { "_id" : 44 }
   ```

2. 멱등성을 가진다.   
   $inc같은 이전 값을 기준으로 변경하는 쿼리를 송신해도 몇 번을 실행해도 동일한 값이 나올 수 있게
   고정으로 지정된 값이 들어가있다.   
   ex)
   ```
   // 원래 값 {data : 1}
    {$inc:{data: 4}} => {$set:{ data: 5}}
   ```

## Primary를 선정하는 방식
Mongodb는 RAFT 방식을 이용하여 primary node를 선출하게 된다.
한번에 한 개의 primary node가 선출되며 아래의 특징이 있다.
- 대다수의 노드와 통신할 수 있어야 한다.
- 가장 최신의 정보를 갖고 있어야한다.
- 지리적으로 다른 노드들과 가까워야 한다.

primary가 셧다운 되었을 때 새로운 primary를 선출하는 과정은 아래와 같다.

- 최근에 primary로부터 동기화 알림이 오지 않음
- 자신을 포함한 대다수의 secondary에 연결 확인이 됨
- 이렇게 알아차린 secondary node는 투표를 할 수 있으며 자격이 있는 경우 자신을 primary로 제안함
- 자신의 최신 동기화 시간과 투표 이후로의 시간을 알림, 이는 각 투표로 부터 시간이 지날때마다 1씩 증가함
- 다른 노드는 투표를 하게 되는데 primary를 제외한 secondary node들만 투표를 하며 후보의 최근 동기화 시간이 더 최신인 노드에 표를 던짐
- 이러한 절차로 특정 노드가 표를 많이 받으면 secondary에서 primary로 변경됨
- 온보딩 프로세스를 진행한 후 쓰기 작업을 받기 시작한다.

이렇게 primary가 된 노드는 아래와 같은 절차를 거친다
- 연결 할 수 있는 secondary 노드 중에서 더 최신 작업 시간을 가진 secondary 가 있는지 확인한 뒤 해당 secondary 로부터 트랜잭션을 복사한다.
- 자신보다 동기화가 덜 된 secondary가 있는지 확인한다. 만약에 있다면 최신 트랜잭션의 이전 및 전달을 진행한다.

이러한 모든 과정은 일반적으로 매우 짧은 시간 내에 이루어지며
이는 다수의 노드가 작성하고 확인한 모든 작업이 손실되지 않도록 보장한다.

## Read concerns and preferences
Replica Set이되면 secondary에서도 데이터를 읽어 올 수 있다. 문제는 secondary에서 데이터를 읽는 시점에서
해당 secondary에 primary node의 데이터가 동기화가 완료되지 않은 시점일 수 있는데 이러한 경우 여러가지 설정을
통해 성능과 정확성 사이에서 적절한 합의점을 찾을 수 있다.

이러한 읽기 설정은 아래와 같다.
- Read Local: 내가 읽은 Replica Set의 node가 가진 최신 정보
- Read Majority: 다수가 가진 최신 정보, 대다수에 의해 활성화된 커밋 지점에 따라 결정
- Read Snapshot: 쿼리가 시작 시점에 존재하는 데이터. 이는 쿼리가 진행되는 동안 발생하는 변경 사항은 반영하지 않음. 하지만 이 과정에서 데이터는 유지됨.
- Read Linearizable: 다수가 내 쿼리 시간에 따라 따라잡을 때까지 기다린 뒤에 반환.

속도가 가장 중요하다면 Local 설정, 신뢰도가 중요하다면 Majority 설정을 쓰면 되겠다.


# 참고자료
- [MongoDB 공식 문서](https://www.mongodb.com/ko-kr/docs/manual/core/document/)

