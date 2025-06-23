---
title: Database - Checkpoint
author: blakewoo
date: 2025-6-21 21:00:00 +0900
categories: [Database]
tags: [Database, DBMS , Recovery, Logging, Checkpoint]
render_with_liquid: false
use_math: true
---

# Recovery - Checkpoint

## 1. 개요
전체 WAL에 대해서 복구를 시도하면 너무나 양이 많기 때문에 체크포인트를 찍어서 이후부터 작업할 수 있게 한다고 했다.   
그리고 이 체크포인트를 만들기 위해서는 여러가지 방법이 있다.

## 2. Non-Fuzzy checkpoints
앞서 Loggin에서 설명했던 체크 포인트 방식이다.

- 새로운 transaction들의 실행을 막는다.
- 현재 active한 transaction들을 모두 완료한 다음, buffer pool에 있는 dirty page들을 disk에 쓰기한다.   
  런타임 성능에 좋지 않지만 복구는 용이해진다.
  
새로운 transaction의 실행을 막는 것부터 이미 성능에 매우 좋지 않다. 사실상 서비스를 중단하는 것과 마찬가지이기 때문이다.   
때문에 이보다 좀 더 나은 방법이 있다.

## 3. Better checkpoints
현재 active한 transaction들이 끝날때까지 기다리는게 아닌 일단 checkpoint는 시작을 하고 이미 진행중인 transaction들을 잠시
멈추게하는 방식으로 checkpoint들을 좀 더 빠르게 진행할 수 있다.
(새로운 transaction 실행을 막고 있는 것은 동일하다)

다만 이 경우 새로운 문제를 야기할 수 있는데, 실질적으로 진행된 Transaction과 변경된 page가 달라 메모리와 disk에 있는
page간에 일관성이 깨질 수 있다.

따라서 이 문제를 해결하기 위해 도입된 두 가지 변수가 있다.

### Active Transaction Table (ATT)
현재 Active 중인 Transaction의 정보를 담고 있다.
- txnId : 트랜잭션 ID
- status : Running, Commit, Candidate for Undo
- lastLSN : 가장 최근에 트랜잭션에서 만들어진 

### Dirty Page Table (DPT)
Buffer Pool에서 디스크로 플러시되지 않은 변경 사항이 포함된 페이지를 추적할때 사용한다. 
Buffer Pool의 Dirty Page당 하나의 항목을 갖고 있는데 페이지를 처음으로 더티하게 만든 로그 레코드의 LSN(recLSN)이다.

위 두 값들이 포함된 예시는 아래와 같다.

![img.png](img.png)



## 4. Fuzzy checkpoints




> ※ 미비된 내용은 추가 업데이트 예정 및 검증 예정이다.
{: .prompt-tip }

# 참고자료
- 서강대학교 김영재 교수님 고급데이터 베이스 강의 자료
- Avi Silberschatz, Henry F. Korth, S. Sudarshan, Database System Concepts(McGraw-Hill, March 2019)
