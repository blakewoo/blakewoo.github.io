---
title: Database - Concurrency control - MVCC(Multi-Version Concurrency Control)
author: blakewoo
date: 2025-6-15 16:00:00 +0900
categories: [Database]
tags: [Database, DBMS ,Transaction, MVCC]
render_with_liquid: false
use_math: true
---

# MVCC(Multi-Version Concurrency Control)

## 1. 개요
2 Phase Locking이나 Optimistic Concurrency Control과 같이 동시성 제어 프로토콜은 아니다.  
되려 이는 여러버전을 유지하기 위해 동시에 트랜잭션이 실행되는 시스템 설계법에 가깝다.
각 데이터 항목의 여러버전을 유지함으로써 여러 트랜잭션이 충돌 없이 동시에 동일한 데이터에 엑세스할 수 있도록 한다.

## 2. 동작 메커니즘
### 1) 다수의 버전
데이터가 각 업데이트 될때마다 과거의 것을 덮어쓰기 하는대신에 새로운 버전이 생성되어야한다.

### 2) 트랜잭션 Timestamp
모든 트랜잭션은 시작할때 고유한 Timestamp를 가져야한다.

### 3) 격리된 스냅샷
트랜잭션은 시작 Timestamp를 기반으로 데이터베이스에서 일관된 스냅샷을 봐야한다.

## 3. 쓰기 읽기 규칙
### 1) 읽기 규칙
트랜잭션이 시작할때 그 시점을 기준으로 database에서 상태를 가져와서 저장한다. (Snapshot을 찍는다고 말한다) 
그렇게 찍어둔 Snapshot은 트랜잭션이 시작된 시점까지 완료된(Commit) 버전의 데이터만 반영하고 있으므로
그 시점까지 완료된 데이터만 읽을 수 있다.

### 2) 쓰기 규칙
트랜잭션 i가 시작할때 그 시점을 기준으로 Snapshot을 찍었다고 해보자.
그렇게 찍어둔 Snapshot에서는 트랜잭션이 시작된 시점까지 완료된(Commit) 버전의 데이터만을 가지고 해당 데이터에
Write를 하게 되는데 만약 트랜잭션 database에 Snapshot 이후 시점보다 더 후에 다른 트랜잭션에서 변경 또는 커밋한 기록이 있다면
충돌(conflict)이 생기므로 트랜잭션 i는 abort하거나 다시 시작해야한다.

### 3) 예시

![img.png](/assets/blog/database/mvcc/img.png)

1) T1의 Timestamp가 1로 세팅되며 (TS(T1)=1) 트랜잭션 상태표에 같이 기재된다.   
   <table>
    <tr>
        <td>TxnId</td>
        <td>Timestamp</td>
        <td>Status</td>
    </tr>
    <tr>
        <td>1</td>
        <td>1</td>
        <td>active</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
    </tr>
   </table>


2) Database에 기본적으로 A의 초기버전인 A0가 있고 T1에서 R(A)를 할때 해당 값을 가져와 읽는다.
   

3) A0의 그다음 버전인 A1을 만들어 값을 넣고, Begin은 1(트랜잭션 1의 Timestamp 값)을 넣고 A0의 END를 1로 세팅한다.         
    <table>
        <tr>
            <td>Version</td>
            <td>Value</td>
            <td>Begin</td>
            <td>End</td>
        </tr>
        <tr>
            <td>A0</td>
            <td>123</td>
            <td>0</td>
            <td>1</td>
        </tr>
        <tr>
            <td>A1</td>
            <td>456</td>
            <td>1</td>
            <td></td>
        </tr>
    </table>

    또한 트랜잭션 상태표에 T2 내용을 기재한다.   
   
    <table>
    <tr>
        <td>TxnId</td>
        <td>Timestamp</td>
        <td>Status</td>
    </tr>
    <tr>
        <td>1</td>
        <td>1</td>
        <td>active</td>
    </tr>
    <tr>
        <td>2</td>
        <td>2</td>
        <td>active</td>
    </tr>
    </table>
   

4) T2에서 READ(A)가 발생했는데 A0를 읽어야한다. 아직 A1가 commit이 안되었기 때문이다.

5) T2에서 Write(A)는 T1이 COMMIT되기 전까지 기다려야한다.

6) T1은 자신이 이전에 쓴 A1을 읽느낟.

7) T1이 끝나고 트랜잭션 상태표에 반영된다.     
   <table>
    <tr>
        <td>TxnId</td>
        <td>Timestamp</td>
        <td>Status</td>
    </tr>
    <tr>
        <td>1</td>
        <td>1</td>
        <td>committed</td>
    </tr>
    <tr>
        <td>2</td>
        <td>2</td>
        <td>active</td>
    </tr>
    </table>

   T2의 W(A)가 이제야 반영되어 새로운 버전을 만들어내게 된다.   
   <table>
    <tr>
        <td>Version</td>
        <td>Value</td>
        <td>Begin</td>
        <td>End </td>
    </tr>
    <tr>
        <td>A0</td>
        <td>123</td>
        <td>0</td>
        <td>1 </td>
    </tr>
    <tr>
        <td>A1</td>
        <td>456</td>
        <td>1</td>
        <td>2</td>
    </tr>
    <tr>
        <td>A2</td>
        <td>789</td>
        <td>2</td>
        <td></td>
    </tr>
    </table>

8) 트랜잭션 2가 끝나고 트랜잭션 상태표에 반영된다.     
   <table>
    <tr>
        <td>TxnId</td>
        <td>Timestamp</td>
        <td>Status</td>
    </tr>
    <tr>
        <td>1</td>
        <td>1</td>
        <td>committed</td>
    </tr>
    <tr>
        <td>2</td>
        <td>2</td>
        <td>committed</td>
    </tr>
    </table>


## 4. 장단점
### 1) 장점
- 읽기간 blocking이 없다. (읽기 lock이 없기 때문)
- 쓰기와 읽기 간의 높은 동시성
- 높은 일관성

### 2) 단점
- 스토리지 용량을 더 사용한다(다수 버전 유지때문)
- Garbage collection이 필요하다. (추가적인 작업이 필요하기 때문에 가끔 성능을 떨어뜨린다)
- 구현이 훨씬 복잡하다.

## 5. MVCC 디자인 구현 - 동시성 제어 프로토콜
### 1) Timestamp Ordering
트랜잭션에 일련(serial) 순서를 결정하는 타임스탬프를 부여

### 2) Optimistic Concurrency Control
새 버전 생성을 위해 트랜잭션별 개인 작업 공간(private workspace) 사용

### 3) Two-Phase Locking
트랜잭션이 논리적 튜플을 읽거나 쓰기 전에 해당 물리적 버전에 대해 적절한 잠금을 획득

## 6. MVCC 디자인 구현 - Version Storage
### 1) 추가 전용 저장 방식 (Append‑Only Storage)
새 버전이 동일한 테이블 공간에 계속 추가된다.

![img.png](/assets/blog/database/mvcc/img_1.png)

#### a. Oldest-to-Newest(O2N)
새 버전을 버전 체인의 가장 마지막에 붙이는 것 - 최신버전을 찾을때마다 순차 검색 필요

#### b. Newest-to-Oldest(N2O)
새 버전을 버전 체인의 가장 앞에 붙이는 것 - 순차 검색 필요 없음

### 2) 시점 이동 저장 방식 (Time‑Travel Storage)
이전 버전이 별도의 테이블 공간으로 복사됨

![img_1.png](/assets/blog/database/mvcc/img_2.png)

### 3) 델타 저장 방식 (Delta Storage)
수정된 속성의 원본 값이 별도의 델타 레코드 공간에 복사됨

![img_2.png](/assets/blog/database/mvcc/img_3.png)

## 7. MVCC 디자인 구현 - Garbage Collection
### 1) Tuple-level
튜플을 직접 검사하여 이전 버전을 찾는 방식으로 아래의 두 가지로 나눌 수 있다.
#### a. Background Vacuuming
별도의 스레드가 주기적으로 테이블을 스캔하여 회수 가능한 버전을 찾아서 제거한다. 모든 스토리지방식에서 가능하다.

#### b. Cooperative Cleaning
워커 스레드는 버전 체인을 탐색하면서 회수 가능한 버전을 식별하여 제거한다. O2N 구조에서만 가능하다.

### 2) Transaction-level
트랜잭션이 자신이 생성한 이전 버전들을 직접 추적하여, DBMS가 가시성을 판단하기 위해 튜플을 스캔할 필요가 없다.
아래의 그림을 보자.

![img_3.png](/assets/blog/database/mvcc/img_4.png)

각 트랜잭션이 이전에 생성한 버전을 확인하여 이를 직접 처리하게끔 요청하는 것이다.

# 참고자료
- 서강대학교 김영재 교수님 고급데이터 베이스 강의 자료
- Avi Silberschatz, Henry F. Korth, S. Sudarshan, Database System Concepts(McGraw-Hill, March 2019)
