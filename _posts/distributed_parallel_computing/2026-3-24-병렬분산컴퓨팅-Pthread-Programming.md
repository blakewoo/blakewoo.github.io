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
int pthread_cond_signal(pthread_cond_t *cv);
int pthread_cond_broadcast(pthread_cond_t *cv);
int pthread_cond_destroy(pthread_cond_t *cv);
```

위 함수중에서 Signal 함수는 한 개의 Thread만 깨우는 데, broadcast 함수는 다른 대기중인 모든 함수를 깨운다.
Thread가 2개인 경우에는 차이가 없지만, 다수의 Thread인 경우 그 다음 실행할 Thread를 정할 주체를 정하는데 도움이 될 수 있다.   
이는 Thread의 우선순위가 정해져있고 해당 Thread를 지정하여 Signal로 깨운다면 구현하는 쪽에서 깨우는 순서를 정하게 되겠지만
그냥 broadcast로 깨운다면 OS에게 맡기는 것이기 때문이다.
만약 wait 중인 Thread가 없을 때 Signal가 호출되었다고 해보자. 이 경우 어디 저장되는게 아니라 무시 된다.

또한 위 함수중에 wait은 mutex_lock과 함께 같이 사용하게 된다. 예를 들어 설명해겠다.
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

##### ※ Mesa semantic & Hoare semantic
Thread A, B가 있다고 해보자. 의사 코드는 아래와 같다.
```c
//Thread A
lock();
if(!condition) wait(); // --- (2)
unlock();

//Thread B
lock();
signal();
//... (1)
unlock();
```

Thread B에서 Critical section을 통과하여 signal()에 도착했을 때 취할 수 있는 방식은 두 가지가 있다.   
Signal 전송 후 아래 (1) 코드를 실행하거나, 혹은 Thread B를 멈추고 Thread A에게 제어권을 넘겨주는 방식이다.   
이를 각각 Mesa semantic, Hoare semantic이라고 한다.

- Mesa semantic   
  (1) 코드를 그대로 실행하는 방식이다. Thread B 이후에 Thread A가 실행될 것이므로 이 경우 Context switching이 1번이라 Context switching overhead가 적다.   
  하지만, 위와 같은 2개의 Thread가 아닌 다수의 Thread의 경우 대기 중인 Thread A가 다시 실행되는 시점에서 실행 조건이 충족되지 않을 수도 있다(condition 변경 가능).
  따라서 신호를 받는 시점에서 조건을 한번 더 확인해야하므로 코드 (2) 부분의 if는 while로 변경되어야한다.
  
- Hoare semantic   
  Thread B가 제어권을 포기하고 Signal을 받는 Thread A에게 제어권을 넘긴다. 이 경우 Mesa semantic에 비해 Context switching이 한번 더 많이 발생하나 Semantic 흐름으로는
  좀 더 자연스럽고 또한 Thread A 실행 이전에 condition이 바뀌지 않기 때문에 Thread A의 Critical section 접근이 보장된다.


일반적으로는 Mesa semantic 방식으로 대부분 구현 되어있다.

#### d. Semaphore
정수 값을 사용하여 여러 스레드의 접근을 제어한다. 만약 0 또는 1이라면 이진 세마포어라고 하며 본질적으로 mutex와 다르지 않다.   
하지만 이 정수값이 1을 넘어가는 숫자일 경우 카운트 세마포어라고 하며 한정된 소수의 자원의 Critical section을 제어하는 방식이 된다.
이 세마포어는 실질적으로 Pthread 표준은 아니고, OS에서 제공하는 기능이다. 따라서 사용시 <semaphore.h>를 별도로 포함해야 한다

일반적으로는 기본적으로 아래 변수를 만들고 시작한다.

```c
sem_t semaphore;
```

위와 같이 만든 변수를 가지고 아래와 같은 함수로 제어한다.

```c
init sem_init(sem_t *sem, int pshared, unsigned int value); // 세마포어 초기화
init sem_destroy(sem_t *sem); // 세마포어 제거
int sem_wait(sem_t *sem); // 세마포어 카운트 하나 줄이기
int sem_post(sem_t *sem); // 세마포어 카운트 하나 늘리기
```

sem_wait 함수의 경우 count 값 체크 후 count를 줄이거나, 줄이고나서 체크하거나 두 가지 방식이 있다.
이는 전적으로 어떻게 구현하느냐에 따라 달린 것이다.

sem_post 함수는 count 값을 증가시킨다. 앞서 condition variable에서 Signal과는 달리 count값이 보존되므로 무시되지 않고 
해당 count 값을 가지고 처리하게 된다.   
앞서 condition variable과 mutex로 구현했던 예제를 세마포어로 구현하면 아래와 같다.

```
// producer
int in = 0;
while(1) {
  sem_wait(&empty);
  sem_wait(&mutex);
  buf[in] = getChar();
  in = (in + 1)%MAX_SIZE;
  sem_post(&mutex);
  sem_post(&full);
}

// consumer
int out = 0;
while(1) {
  sem_wait(&full);
  sem_wait(&mutex);
  useChar(buf[out]);
  out = (out + 1)%MAX_SIZE;
  sem_post(&mutex);
  sem_post(&empty);
}
```

full은 세마포어 카운트 0으로 초기화되고, empty는 MAX_SIZE로 초기화 된다.   
즉 full은 꽉 찼는지, empty는 빈 자리가 몇개나 있는지 나타내는 것이다.

만약 sem_wait의 순서를 뒤바꾸면 어떻게 될까?
원래   sem_wait(&empty) -> sem_wait(&mutex) 였던 것을
sem_wait(&mutex) -> sem_wait(&empty)로 바꾼다면? 당연하지만 dead lock이 발생한다.   
condition variable 없이 mutex를 사용한 것과 동일한 문제가 발생하는 것이다.

### 5) Read-Write lock
만약 어떤 연결 리스트에 접근한다고 생각해보자. 위에서 배운 바와 같이 연결 리스트 전체에 대해서 lock을 걸거나 혹은 접근하고자 하는 Node의
앞 뒤에 대해서만 lock을 건다고 가정해보자. 이 경우 사실 READ의 경우 몇 개의 Thread가 동작해도 원 데이터에는 아무런 변화가 없는데
lock 때문에 접근을 못한다고 하면 너무 비효율적이다. 따라서 Pthread는 readlock과 writelock을 지원한다.

```c
pthread_rwlock_rdlock(&rwlock);
pthread_rwlock_wrlock(&rwlock);
pthread_rwlock_unlock(&rwlock);
```

readlock과 writelock의 호환성은 아래와 같다.

<table>
    <tr>
        <td>Current / New</td>
        <td>Read Lock</td>
        <td>Write Lock</td>
    </tr>
    <tr>
        <td>Read Lock</td>
        <td>O</td>
        <td>X</td>
    </tr>
    <tr>
        <td>Write Lock</td>
        <td>X</td>
        <td>X</td>
    </tr>
</table>

#### a. 구현 옵션
이 과정에서 스레드 간의 우선 순위를 어떻게 처리하느냐에 따라 두가지 방식으로 나뉜다.

##### ⓐ 읽기 우선 
기다리는 읽기 Thread가 있다면 쓰기 Thread보다 먼저 실행되게 하며 읽기 Thread가 대기하지 않게 하는 방식이다.   
아래의 예시를 보자. 요청이 ``READ_1 - WRITE_1 - READ_2 - WRITE_2``` 순으로 들어왔다고 가정해보자.
(언더바는 설명의 편의를 위해 임의로 붙인 것이고 실제로는 READ - WRITE - READ - WRITE 라고 생각하면 된다)
이를 위해 정의된 변수는 아래와 같다.

```c
semaphore mutex = 1, wrt=1;
int readcount = 0;
```

Writer Process는 아래와 같다.
```c
do {
  wait(wrt);
  ...
  writing is performed
  ...
  signal(wrt)

} while(1);
```

Reader Process는 아래와 같다.
```c
do {
 wait(mutex);
 readcount ++;
 if(readcount == 1) wait(wrt);
 signal(mutex);
 ...reading is performed ...
 wait(mutex)
 readcount--;
 if(readcount == 0) signal(wrt);
 signal(mutex);
} while(1);
```

1. READ_1 도착    
READ_1이 도착하면 wait(mutex)를 거쳐 mutex lock을 획득하고, read_count가 1이기 때문에 wrt lock도 획득하게된다.

2. WRITE_1 도착
WRITE_1이 도착하면 READ_1이 wrt lock을 가지고 있기 때문에 wait(wrt)에서 대기하게 된다.

3. READ_2 도착
READ_2가 도착하면 READ_1이 mutex lock을 반환하기 전까지 wait(mutex)에 대기한다.
   
4. WRITE_2 도착
WRITE_2가 도착하면 READ_1의 wrt lock 반환을 대기해야하므로 wat(wrt)에 대기한다.
   
이후 READ_1이 signal(mutex)를 지나면 READ_2가 실행 가능하므로 READ_1 -> READ_2 순으로 실행되며, WRITE_1과 WRITE_2는 
WRITE_1 -> WRITE_2 순으로 실행되든, WRITE_2 -> WRITE_1 순으로 실행되든 한다. (Signal 시 어느 Thread를 깨우는지는 불명확)


###### ※ 예시가 WRITE - READ - WRITE - READ 일때
1. WRITE_1 도착   
Writer process의 wait(wrt)를 호출하여 lock을 획득하고 작업을 시작한다.
   
2. READ_1 도착   
Reader process의 wait(mutex)를 호출하여 lock을 획득 후 readcount를 1로 만들어놓는데, if에 걸려서 
wait(wrt)를 호출한다. 이 과정에서 Write_1이 wrt lock을 갖고 있기 때문에 대기 한다.

3. WRITE_2 도착
Writer process의 wait(wrt)를 호출하여 wrt lock을 획득하려하는데, WRITE_1이 갖고 있어서 wait(wrt)에 걸려있다.
   
4. READ_2 도착    
Reader process의 wait(mutex)를 호출하는데 mutex lock은 READ_1이 가지고 있다. wait(mutex)에서 걸려있다.

여기까지 진행되는데, 만약에 WRITE_1이 완료되어 Signal(wrt)를 호출 하고 종료되었다고 해보자. 이후 순서는 아래와 같다.

1. READ_1 or WRITE_2 실행      
왜 둘 중 하나가 실행되냐고 묻는다면, 앞서 설명했듯이 Mutex든 Semaphore든 깨우는 Thread가 무엇일지는 알 수가 없기 때문이다. 
둘다 wait(wrt)에 걸려있었기 때문에 둘 중에 하나가 선택되어 실행된다.
   

2. READ_1 먼저 실행될 경우 다음은 READ_2 -> WRITE_2 실행   
READ_1이 먼저 실행되면 wrt lock을 가지고 Signal(mutex)를 거치기 때문에 wait(mutex)에 걸려있던 READ_2는 mutex lock을 가지고 내려올 수 있다.
이 과정에서 readcount는 이미 2이므로 wait(wrt)에 걸리지 않는다. 이후 READ_2까지 readcount--를 지나게 되면 readcount가 0이기 때문에 signal(wrt)를 지나게 되고
비로소 WRITE_2가 wrt lock을 가지고 실행하게 된다.


3.  WRITE_2 먼저 실행될 경우 다음은 READ_1 -> READ_2 실행   
WRITE_2가 실행될 경우 WRITE_2가 signal(wrt)를 지날때까지 READ_1은 wait(wrt)에 READ_2는 wait(mutex)에 걸려있다.   
WRITE_2가 완료되면 READ_1이 풀려나고 계속 진행되면서 signal(mutex)를 지나기 때문에 READ_2도 풀리면서 순차적으로 실행된다.

##### ⓑ 쓰기 우선
쓰기 Thread가 대기중이라면 새로운 읽기 Thread의 접근을 막고 쓰기 Thread 작업을 먼저 처리한다. 대부분의 현대적인 시스템에서는 쓰기 우선 방식을 많이 사용한다.
읽기 우선 방식과 동일한 예시를 사용하겠다. 요청이 ``READ_1 - WRITE_1 - READ_2 - WRITE_2``` 순으로 들어왔다고 가정해보자.
(이번에도 언더바는 설명의 편의를 위해 임의로 붙인 것이고 실제로는 READ - WRITE - READ - WRITE 라고 생각하면 된다)
이를 위해 정의된 변수는 아래와 같다.

```c
int readcount = 0, writecount = 0;
semaphore mutex1 = 1, mutex2 = 1, rd = 1,wrt = 1;
```

Writer Process는 아래와 같다.
```c
do {
 wait(mutex2);
 writecount ++;
 if(writecount == 1) wait(rd);
 signal(mutex2);
 wait(wrt);
 // ... write is performed
 signal(wrt);
 wait(mutex2);
 writecount --;
 if(writecount == 0) signal(rd);
 signal(mutex2);
} while(1);
```

Reader Process는 아래와 같다.
```c
do {
 wait(rd);
 wait(mutex1);
 readcount ++;
 if(readcount == 1) wait(wrt);
 signal(mutex1);
 signal(rd);
 // ... reading is performed
 wait(mutex1);
 readcount --;
 if(readcount == 0) signal(wrt);
 signal(mutex1);
} while(1);
```

처음 도착하는 READ_1이 읽기를 실제로 실행하는 부분 초입까지 도착한 상태라고 해보자.

1. READ_1 도착    
   READ_1이 도착하면 wait(rd)를 거쳐 rd lock을 획득하고, wait(mutex1)를 거쳐 mutex1 lock을 획득하고
   readcount를 증가시키는데 1이기 때문에 wrt lock도 획득하게된다. 이후 mutex1 lock을 해제하고, rd lock도 반환한다.

2. WRITE_1 도착
   WRITE_1이 도착하면 wait(mutex2)를 거쳐 mutex2 lock을 획득하고 writecount가 1이기 때문에 wait(rd)에 걸려 있다가 READ_1이 signal(rd)를
   지나면 wait(wrt)에 걸려있다. READ_1이 읽기를 하지 않았기 때문에 wrt lock은 READ_1이 갖고 있기 때문이다.

3. READ_2 도착
   READ_2가 도착하면 wait(rd)에 걸려있는데 WRITE_1이 rd lock을 갖고 있기 때문이다

4. WRITE_2 도착
   WRITE_2가 도착하면 READ_1이 wrt lock을 가진채로 읽기 작업 중이라 wait(wrt)에 걸려있다.
   
이후 READ_1이 signal(mutex1)을 거쳐 모두 종료되었다고 하면, READ_1이 signal(wrt)를 지났을 것이므로 WRITE_1 또는 WRITE_2가 실행된다.

1. WRITE_1이 실행될 경우 WRITE_2 -> READ_2 실행
2. WRITE_2가 실행될 경우 WRITE_1 -> READ_2 실행

이는 마지막 WRITE Thread 작업이 rd lock을 갖고 있기 때문에 이를 풀어주지 않으면 READ 작업은 할 수가 없기 때문에 WRITE의 모든 작업이 끝난뒤 READ Thread가 작동하게 된다.

### 6) Thread-Safety
다수의 Thread로 호출했을 때 문제가 없다면 Thread-Safety하다고 부른다.   
가령 문자열을 받아들여 공백과 같은 구분자로 분리하여 토큰화하는 strtok 함수가 있다고 해보자.
이 strstok은 이제까지 어디까지 잘랐는지 저장하는 pointer가 있다.   
이 경우 global 변수로 둘 수 있는데, 만약 다수의 Thread를 이용해서 해당 Pointer를 변경할 일이 있다고 하면   
Thread-Safety하지 못한 함수라고 할 수 있다.

그외 Unsafe한 C library 함수들의 예시는 아래와 같다.
- stdlib.h : rand
- time.h : localtime


# 참고자료
- 서강대학교 박성용 교수님 강의자료 - 병렬 분산 컴퓨팅  

# 원문 참고자료들
- Peter S. Pacheco, An Introduction to Parallel Programming,  Elsevier Inc. (Morgan Kaufmann), 2011, ISBN 978-0-12-374260-5
- Gerassimos Barlas, Multicore and GPU Programming – An Integrated Approach, Elsevier Inc. (Morgan Kaufmann), 2015, ISBN 978-0-12-417137-4.
- G. Coulouria, J. Dollimore, T. Kindberg, and G. Blair, Distributed Systems: Concepts and Design, 5 th Edition, Pearson, 2012, ISBN 978-0-273-76059-7
- M. van Steen and A. S. Tanenbaum, Distributed Systems, 3 rd Edition, 2017
- Martin Kleppmann, Designing Data-Intensive Applications, 1 st Edition, O'Reilly Media, 2017, ISBN 978-1491903070 (또는 2nd  Edition in February 2026)
