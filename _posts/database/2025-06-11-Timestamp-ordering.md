---
title: Database - Timestamp ordering
author: blakewoo
date: 2025-6-11 23:30:00 +0900
categories: [Database]
tags: [Database, DBMS ,Transaction]
render_with_liquid: false
use_math: true
---

# Database - Timestamp ordering

## 1. 개요
Lock 없이 Timestamp만을 가지고 직렬화를 판단하는 방식을 말한다.   
여기서 Timestamp는 system clock이 될수도, 논리적카운터가 될수도 혹은 그 외의 방식으로 구현된 시간 순에 따라
생성되는 고유한 값이다

## 2. Basic Timestamp Ordering
각 트랜잭션의 Timestamp을 기준으로 Rollback 할지 아니면 Commit 할지 정하는 기본적인 방식으로
어떤 자원 X에 대한 READ 트랜잭션은 Read-TS(X), 어떤 자원 X에 대한 Write 트랜잭션은 Write-TS(X)라고 할 때
Read write 둘다 마지막 연산을 기준으로 Timestamp를 찍는다.

### 1) 장점
- 직렬화가 보장된다.
- lock이 없다. dead lock을 방지할 수 있으며, lock overhead가 없다.

### 2) 단점
- 잦은 abort가 일어난다. 특히 오래된 트랜잭션에 대해서
- 연쇄 abort가 가능하다. 이렇게 되면 트랜잭션들은 기아 현상을 겪을 수 있다.

### Interleaved Execution Anomalies
## 2-1. Thomas Write Rule
## 3. Optimistic Concurrency Control
> ※ 추가 업데이트 및 검증 예정이고, 아직 완벽하지 않으니 참고만 바란다.
{: .prompt-tip }


# 참고자료
- 서강대학교 김영재 교수님 고급데이터 베이스 강의 자료
