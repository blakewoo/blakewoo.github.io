---
title: Database - Checkpoint
author: blakewoo
date: 2025-6-24 21:00:00 +0900
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

### 1) Active Transaction Table (ATT)
현재 Active 중인 Transaction의 정보를 담고 있다.
- txnId : 트랜잭션 ID
- status : Running, Commit, Candidate for Undo
- lastLSN : 가장 최근에 트랜잭션에서 만들어진 

### 2) Dirty Page Table (DPT)
Buffer Pool에서 디스크로 플러시되지 않은 변경 사항이 포함된 페이지를 추적할때 사용한다. 
Buffer Pool의 Dirty Page당 하나의 항목을 갖고 있는데 페이지를 처음으로 더티하게 만든 로그 레코드의 LSN(recLSN)이다.
체크 포인트에는 어떤 Page가 Dirty한지 표기한다.

### 3) 예시
ATT와 DPT가 포함된 예시는 아래와 같다.

![img.png](/assets/blog/database/checkpoint/img.png)

체크 포인트에 ATT가 Active Transaction Table이고, ATT={T2} 라는 식으로 표기되어있다.
DPT가 Dirty Page Table이며 DPT={P22}라는 식으로 표기되어있다.

각각 T2가 체크포인트 시점에서 아직 끝나지 않았었고, 페이지 22가 DISK로 FLUSH 되지 않았다고 표기하고 있다.

물론 이러한 방식의 check point는 Non-Fuzzy checkpoints한 방식보다는 낫지만 checkpoint간에 stall은 해야하며
이때문에 가장 좋은 방법은 아니다.

## 4. Fuzzy checkpoints
fuzzy checkPoint는 앞선 checkpoint와는 달리 실행을 멈출 필요가 없으며 동적으로 체크포인트를 찍을 수 있는 방식이다.   
하지만 조금 복잡하다.

### 1) 방법
앞선 다른 방식과 달리 fuzzy checkpoint는 CHECKPOINT-BEGIN과 CHECKPOINT-END를 갖는다.  
이 값 역시 LOG에 모두 기재되며, CHECKPOINT-BEGIN은 CHECKPOINT의 시작만 알릴뿐이고, CHECKPOINT-END는 
체크포인트가 종료된 시점에서 ATT와 DPT를 기재한다.

### 2) 복구 방식
이 Fuzzy checkpoint는 3단계의 복구 페이즈가 필요한데 각각 아래와 같다.

#### a. Analysis
WAL을 MasterRecord에 기재된 시점부터 쭉 살피고 어느시점에서 어떤 page가 dirty한지, 실행중인 트랜잭션은 어떤 것인지
모두 분석한다.

#### b. Redo
모든 작업을 다시 재 시작한다.

#### c. Undo
크래시가 일어난 시점부터 역으로 올라가면서 각 작업들을 Undo한다.

> ※ 미비된 내용은 추가 업데이트 예정 및 검증 예정이다.
{: .prompt-tip }

# 참고자료
- 서강대학교 김영재 교수님 고급데이터 베이스 강의 자료
- Avi Silberschatz, Henry F. Korth, S. Sudarshan, Database System Concepts(McGraw-Hill, March 2019)
