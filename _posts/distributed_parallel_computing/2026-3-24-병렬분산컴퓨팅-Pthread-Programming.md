---
title: 병렬분산컴퓨팅 - Pthread programming
author: blakewoo
date: 2026-3-26 21:00:00 +0900
categories: [Distributed & parallel computing]
tags: [Parallel computing, Distributed computing] 
render_with_liquid: false
use_math: true
---

# 병렬 분산 컴퓨팅 - Pthread programming
## 1. Pthread programming
### 1) Posix(POSIX는 Portable Operating System Interface for Unix) Threads
그냥 흔히들 Pthread라고 불리는 thread이다. C언어 프로그램과 링크하여 사용할 수 있는 라이브러리형태로 제공되며, 멀티스레드 프로그래밍을 위한 다양한 함수를 정의하고 있다.   
컴파일시에는 보통 -lpthread 옵션을 추가하여 라이브러리를 링크해야한다.

### 2) 주요 스레드 관리 함수 
#### a. pthread_create   
새로운 스레드를 생성하고 지정된 함수(runner)를 실행하게 한다.   
프로세스 모델로 치면 fork() 후에 exec()를 실행한 것이다.
  
#### b. pthread_exit   
실행 중인 스레드를 종료하며, 부모 스레드에게 작업이 완료되었음을 알린다.   
프로세스 모델로 치면 exit()이다.

#### c. pthread_join   
부모 스레드가 특정 자식 스레드가 종료될 때까지 기다리는 동기화 함수이다.   
프로세스 모델로 치면 wait()이다.

#### d. pthread_detach   
스레드를 독립적인 데몬(Daemon) 상태로 만들어, 부모 스레드가 기다릴 필요없이 종료 시 자원이 자동으로 회수되게 한다.

#### ※ 참고
프로세스의 경우라면 부모 프로세스가 자식 프로세스를 종료해주어야 종료된다. 만약 대상 프로세스가 background에서 구동된다면   
별도의 Signal을 발생시켜 종료되며, 자식 프로세스를 두고 부모 프로세스가 종료되어버리면 자식 프로세스는 고아 프로세스가 되어 init이 처리하게 된다.

### 3) 스레드 프로그래밍 모델
#### a. Master-Worker Models
마스터 스레드가 요청을 받고, 각 요청에 대해 워커 스레드를 생성하여 작업을 할당하는 방식이다.    
이 경우 요청 후 스레드를 생성하기까지 오버헤드가 있기 때문에, 미리 Thread를 일정 크기만큼 생성해 두고 이를 Pool로 두어 각각 할당시키는 방식을 많이 사용한다.

#### b. Peer Models
모든 스레드가 대등한 관계에서 시작 신호를 기다렸다가 각자의 작업을 수행하고 모두 마칠 때까지 기다리는 방식이다.

#### c. Pipeline Models
작업을 여러 단계(stage)로 나누어, 한 스레드의 출력 결과가 다음 스레드의 입력이 되는 연쇄적인 구조이다.

### 4) 동기화 매커니즘
#### a. Mutex lock
상호 배제(Mutual Exclusion)를 위한 가장 기본적인 도구로, 임계 영역(Critical Section)에 오직 하나의 스레드만 접근할 수 있도록 잠금 기능을 수행한다.   
기본적으로 아래 변수를 만들고 시작한다.
```c
pthread_mutex_t mutex;
```

사용 방식은 아래와 같다.

```c
int ptrehad_mutex_init(pthread_mutext_t *mp, const pthread_mutexattr_t *mattr);
int pthread_mutex_destroy(pthread_mutex_t *mp);
int pthread_mutex_lock(pthread_mutex_t *mp);
int pthread_mutex_unlock(pthread_mutex_t *mp);
```

위에서부터 init, destroy, lock, unlock이다.
init과 destroy는 먼저 lock을 생성하고 없애는 함수이다.
lock 함수는 말 그대로 lock을 거는 함수이다. 이 lock은 기본적으로 blocking 함수로 저 함수를 호출하면 lock을 받기 전까지는
그 함수 아래로 실행이 안된다. 이를 방지하기 위해 아래의 함수가 있다.

```c
int pthread_mutex_trylock(pthread_mutex_t *mp);
```

위 함수는 당장 lock을 받아올 수 없다면 EBUSY라는 에러 코드를 반환하고 해당 함수 아래의 코드로 넘어가서 실행해버린다.

그 다음 함수는 unlock에 대한 내용인데 주의 깊게 봐야할 부분이 있다. 이 unlock이 호출될 경우 lock을 유지하는 스레드들에
우선 순위가 있다면 우선순위대로 lock을 받게 되지만, 아니라면 정말 무작위로 lock을 갖게 된다.

#### b. Spin lock
짧은 시간 동안 락을 보유할 것으로 예상될 때 사용하며, 컨텍스트 스위칭 없이 CPU를 점유한 채 대기(busy waiting)하는 방식이다.
기본적으로 아래 변수를 만들고 시작한다.
```c
pthread_spinlock_t lock;
```

위와 같이 만들어진 변수는 아래의 함수를 통해 제어한다.

```c
int ptrehad_spin_init(pthread_mutext_t *lock, int pshared);
int pthread_spin_destroy(pthread_mutex_t *lock);
int pthread_spin_lock(pthread_mutex_t *lock);
int pthread_spin_trylock(pthread_mutex_t *lock);
int pthread_spin_unlock(pthread_mutex_t *lock);
```

기본적으로 CPU를 점유하기 때문에 시간이 길어지면 좋지 않으나, 짧은 시간 내에 멀티 프로세서 환경이라면 Context switching 횟수를 
줄이는 효과가 있기 때문에 Context switching overhead가 줄어들어 성능이 향상된다.

#### c. Condition Variable (조건 변수)
특정 조건이 참이 될 때까지 스레드를 대기 상태로 두고, 조건이 충족되면 다른 스레드가 신호를 보내 깨워주는 방식이다.   
일반적으로는 mutex_lock과 같이 사용하며 기본적으로 아래 변수를 만들고 시작한다.
```c
pthread_cond_t condition;
```

위와 같이 만들어진 변수는 아래의 함수를 통해 제어한다.

```c
int pthread_cond_init(pthread_cond_t *cv, const pthread_condattr_t *cattr);
int pthread_cond_destroy(pthread_cond_t *cv);
int pthread_cond_wait(pthread_cond_t *cv, pthread_mutex_t *mutex);
int pthread_cond_destroy(pthread_cond_t *cv);
```

위 함수중에 wait은 mutex_lock과 함께 같이 사용하게 된다. 예를 들어 설명해겠다.
아래의 코드를 보자.

```c
// producer
while(1) {
  pthreads_mutex_lock(&lock);
  while(count == MAX_SIZE);
  buf[count] = getChar(); count++;
  pthread_mutex_unlock(&lock);
}

// consumer
while(1) {
  pthreads_mutex_lock(&lock);
  while(count == 0);
  useChar(buf[count-1]); count--;
  pthread_mutex_unlock(&lock);
}
```

위 코드의 경우 문제가 있다. 바로 count가 MAX_SIZE가 되버리면 producer에서 탈출할 수가 없다는 점이다.   
이는 lock을 producer에서 갖고 있기 때문에 consumer 쪽에서 critical section에 접근할수가 없고 이로 인해 count가 줄어들수가 없다.
이렇게되면 계속 producer 안에서 갖히게 되는 것이다. 이러한 문제를 방지하기 위해 condition variable을 사용한다.   
이 함수를 쓰면 consumer 측에서 lock을 접근 할때 잠시 lock을 해제해주어 교착상태를 방지시켜준다.   
사용법은 아래와 같다.

```c
// producer
while(1) {
  pthreads_mutex_lock(&lock);
  while(count == MAX_SIZE) pthread_con_wait(&notFull, &lock);
  buf[count] = getChar(); count++;
  pthread_cond_signal(&notEmpty);
  pthread_mutex_unlock(&lock);
}

// consumer
while(1) {
  pthreads_mutex_lock(&lock);
  while(count == 0) pthread_con_wait(&notEmpty, &lock);
  useChar(buf[count-1]); count--;
  pthread_cond_signal(&notFull);
  pthread_mutex_unlock(&lock);
}
```

#### d. Semaphore
정수 값을 사용하여 여러 스레드의 접근을 제어한다. (단, 세마포어는 엄밀히 Pthreads 표준의 일부는 아니므로 <semaphore.h>를 별도로 포함해야 한다)

### 5) Read-Write lock
### 6) Thread-Safety
> ※ 추가 업데이트 예정
{: .prompt-tip }

# 참고자료
- 서강대학교 박성용 교수님 강의자료 - 병렬 분산 컴퓨팅  

# 원문 참고자료들
- Peter S. Pacheco, An Introduction to Parallel Programming,  Elsevier Inc. (Morgan Kaufmann), 2011, ISBN 978-0-12-374260-5
- Gerassimos Barlas, Multicore and GPU Programming – An Integrated Approach, Elsevier Inc. (Morgan Kaufmann), 2015, ISBN 978-0-12-417137-4.
- G. Coulouria, J. Dollimore, T. Kindberg, and G. Blair, Distributed Systems: Concepts and Design, 5 th Edition, Pearson, 2012, ISBN 978-0-273-76059-7
- M. van Steen and A. S. Tanenbaum, Distributed Systems, 3 rd Edition, 2017
- Martin Kleppmann, Designing Data-Intensive Applications, 1 st Edition, O'Reilly Media, 2017, ISBN 978-1491903070 (또는 2nd  Edition in February 2026)
