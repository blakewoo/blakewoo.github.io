---
title: 병렬분산컴퓨팅 - Parallel software & Performance
author: blakewoo
date: 2026-3-21 18:00:00 +0900
categories: [Distributed & parallel computing]
tags: [Parallel computing, Distributed computing] 
render_with_liquid: false
use_math: true
---

# 병렬 분산 컴퓨팅 - Parallel software & Performance

## 1. Parrallel software
### 1) Programming model
병렬 프로그래밍의 경우 실제로 구동되는 환경과 모델 사이의 추상화 부분에서 좀 상이할 수 있다.   
이게 무슨 말이냐면 프로그래밍적으로는 분산 메모리를 가장하고 만들어질 수 있지만, 실제 구동되는 환경은 공유 메모리 환경일 수 있으며 반대로
공유 메모리를 가장하고 만들어진 프로그램이지만 실제 구동되는 환경을 분산 메모리일 수 있다는 점이다.

#### a. Shared memory model
변수를 공유(shared)하거나 전용(private)으로 설정할 수 있다.
스레드 간의 통신은 공유 변수를 통해 암시적(implicit)으로 이루어진다.
대표적인 예로 OpenMP, Pthreads, Cilk 등이 있다.

#### b. Distributed memory model
메시지 패싱(Message-Passing) 모델이라고도 한다.
각 프로세스는 자신의 전용 메모리에만 직접 접근할 수 있으며,
통신은 MPI(Message Passing Interface)나 소켓과 같은 API를
명시적으로 호출하여 데이터를 주고받는 방식으로 이루어진다.

#### c. Hybrid model
두 가지 이상의 모델을 결합한 방식이다. (예: MPI + OpenMP, MPI + CUDA)

## 2. 병렬성의 유형(Types of Parallelism)
### 1) 데이터 병렬성(Data parallelism)
여러 프로세서가 서로 다른 데이터에 대해 동일한 작업을 수행하는 방식이다.

### 2) 태스크 병렬성(Task parallelism)
여러 프로세서가 동일한 데이터에 대해 서로 다른 작업을 수행하는 방식이다.

## 3. 병렬 소프트웨어 개발시 고려사항
### 1) 비결정성(Non-determinism)
동일한 입력에 대해서도 실행할 때마다 결과가 달라질 수 있는 특성이다.
이는 여러 스레드가 독립적으로 실행되면서 문장을 완료하는 상대적인 속도가 매번 다르기 때문에 발생하며,
경쟁 상태(race condition)의 원인이 된다.

### 2) 스레드 안정성(Thread Safety)
여러 스레드가 동시에 실행하더라도 문제가 발생하지 않는 코드 블록을 의미한다.
예를 들어, 정적 변수를 사용하는 함수는 여러 스레드가 동시에 호출할 경우 안전하지 않을 수 있으며,
rand()나 asctime() 같은 일부 표준 C 라이브러리 함수는 스레드에 안전하지 않아 재진입 가능한(reentrant)
버전(rand_r 등)을 사용해야 한다.

## 3. Performance
### 1) 주요 성능 지표 (Speedup 및 Efficiency)
#### a. 스피드업 (S)
#### b. 효율성 (E)

### 2) 법칙
### Amdahl's law
### Gustafson's law
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
