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
어떤 자원 X에 대한 READ 트랜잭션 $T_{i}$는 $Read-TS_{i}(X)$, 어떤 자원 X에 대한 Write 트랜잭션 $T_{i}$은 $Write-TS_{i}(X)$라고 할 때
Read write 둘다 마지막 연산을 기준으로 Timestamp를 찍는데, 마지막 연산을 각각 Read-TS(X), Write-TS(X)로 표현한다.

### 1) 장점
- 직렬화가 보장된다.
- lock이 없다 (따라서 dead lock을 방지할 수 있으며, lock overhead가 없다)

### 2) 단점
- 잦은 abort가 일어난다. (특히 오래된 트랜잭션에 대해 발생한다)
- 연쇄 abort가 가능하다. (잦아지면 트랜잭션들은 기아 현상을 겪을 수 있다   
  특히 긴 트랜잭션과 짧은 트랜잭션 다수가 있을 경우 연쇄 abort는 더 많을 수 있다)

### 3) 방식 
말로 설명하면 힘드니 수도 코드를 기재하고 하나하나 설명하겠다.

#### a. Read
```c
if( TS(i) < Write-TS(x)) {
  Abort Ti and Rollback/Restart;
}
else {
  Read(x);
  if(Read-TS(x) < TS(Ti))
    Read-TS(x) = TS(Ti);
}
```

현재 실행중인 트랜잭션의 Timestamp가 Write-Timestamp 보다 이전이면 현재 트랜잭션을 Abort하고
Rollback 하고 재 시작해야한다. 예를 들어보자.

![img_2.png](/assets/blog/database/timestamp_ordering/img_2.png)

1) T1에 대해서 TIMESTAMP는 1이다.
2) R-TS의 B가 1로 세팅된다 (B에 대해서 READ가 일어났으므로)
3) T2에 대해서 TIMESTAMP는 2이다.
4) R-TS의 B가 2로 세팅된다 (B에 대해서 READ가 일어났으므로)
5) W-TS의 B가 2로 세팅된다 (B에 대해서 WRITE가 일어났으므로)
6) R-TS의 A가 2로 세팅된다
7) T1의 TIMESTAMP가 T2의 TIMESTAMP보다 이전이니까 ABORT없이 넘어간다.
8) W-TS의 A가 2로 세팅된다.
9) 커밋된다.


#### b. Write
```c
if( TS(i) < Read-TS(x) OR TS(Ti) < Write-TS(x)) {
  Abort Ti and Rollback;
}
else {
  Write(x);
  Write-TS(x) = TS(Ti);
}
```

![img_3.png](/assets/blog/database/timestamp_ordering/img_3.png)

1) T1에 대해서 TIMESTAMP는 1이다.
2) R-TS의 A가 1로 세팅된다 (A에 대해서 READ가 일어났으므로)
3) T2에 대해서 TIMESTAMP는 2이다.
4) W-TS의 A가 2로 세팅된다 (A에 대해서 Write가 일어났으므로)
5) T2가 commit
6) T1의 TIMESTAMP가 W-TS A보다 작기 때문에 위반이 일어나서 ABORT 된다.

#### c. Thomas Write Rule
```c
if( TS(i) < Read-TS(x)) {
  Abort Ti and Rollback;
}
if(TS(Ti) < Write-TS(x)) {
  // 그대로 진행하되 TS(Ti)에서 하는 Write 연산은 실행하지 않음
}
else {
  Write(x);
  Write-TS(x) = TS(Ti);
}
```
위에서 설명한 Write 연산에 대해서 좀 더 최적화된 버전이다.   
어차피 덮어쓸 것이라면 Write 후 Write 시 abort를 해봐야 성능만 깎아먹으니
그냥 전 Write는 실행하지 않고 commit 해버리는 것이다.   

만약 이후에 Read가 등장한다면 어차피 ```if( TS(i) < Read-TS(x)) ``` 조건문에서 걸리니
상관없다는 것이다.

## 3. Optimistic Concurrency Control
### 1) 개요
1981년 카네기 멜론에서 만들어진 방식으로 Conflict가 실제로는 많지 않다는 점에서 착안한 방식이다.

### 2) Phase
1. Read Phase    
   read/write를 위해서 global 데이터에서 private workspace로 복사해온다.
  

2. Validation Phase    
   트랜잭션을 commit할때 다른 트랜잭션과 conflict가 있는지 확인해본다. 만약에 문제가 없다면
   타임스탬프를 그대로 반영한다.
   
  
3. Write Phase    
   Validation이 성공하면 private workspace를 global data를 쓰기를 하고, 실패했다면 abort하고 다시 트랜잭션을 실행한다.

### 3) Validation 방식
Validation을 하는 트랜잭션 기준으로 어느방향으로 보냐에 따라 방식이 두가지로 나뉜다.

- Backward Validation   
  
![img.png](/assets/blog/database/timestamp_ordering/img.png)

- Forward Validation   

![img_1.png](/assets/blog/database/timestamp_ordering/img_1.png)

### 4) 성능 이슈
- 데이터 복사하는데 드는 높은 오버헤드
- Validation/Write Phase에서 드는 병목 현상

> ※ 추가 업데이트 및 검증 예정이고, 아직 완벽하지 않으니 참고만 바란다.
{: .prompt-tip }


# 참고자료
- 서강대학교 김영재 교수님 고급데이터 베이스 강의 자료
