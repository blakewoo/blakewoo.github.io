---
title: 병렬분산컴퓨팅 - Parallel hardware
author: blakewoo
date: 2026-3-19 21:30:00 +0900
categories: [Distributed & parallel computing]
tags: [Parallel computing, Distributed computing] 
render_with_liquid: false
use_math: true
---

# 병렬 분산 컴퓨팅 - Parallel hardware
## 1. 개요
사실 대부분은 컴퓨터 구조 카테고리에 [병렬 처리](https://blakewoo.github.io/posts/%EC%BB%B4%ED%93%A8%ED%84%B0%EA%B5%AC%EC%A1%B0-%EB%B3%91%EB%A0%AC%EC%B2%98%EB%A6%AC/) 및 [병렬 처리 - Cache 일치](https://blakewoo.github.io/posts/%EC%BB%B4%ED%93%A8%ED%84%B0%EA%B5%AC%EC%A1%B0-%EB%B3%91%EB%A0%AC%EC%B2%98%EB%A6%AC-Cache%EC%9D%BC%EC%B9%98/) 포스팅에 포함되어있다.
이번 포스팅에서는 해당 포스팅에서 포함되어있지 않은 내용을 커버해 볼 것이다.

## 2. UMA vs NUMA
### 1) UMA(Uniform Memory Access)
메모리에 접근하는 시간이 동일(Uniform)한 아키텍처이다.   
그림으로 그려보면 아래와 같다.

![img.png](/assets/blog/distributed_parallel_computing/Parallel_hardware/img.png)

### 2) NUMA(Non-Uniform Memory Access)
메모리에 접근하는 시간이 동일하지 않은(Non-Uniform)한 아키텍처이다.
그림으로 그려보면 아래와 같다.

![img_1.png](/assets/blog/distributed_parallel_computing/Parallel_hardware/img_1.png)

위 그림만 보면 둘다 memory까지 걸리는 시간이 같은 것 같은데? 할 수 있겠지만
오해하면 안된다. chip1의 메모리에 chip2 core가 접근할 경우 chip2 memory에 걸리는 시간과 동일하지 않다.

## 3. Interconnection Networks
### 1) Bus Interconnect
버스에 연결된 장치의 수가 늘어날수록 버스 사용을 위한 경합(contention)이 심해지며,
이로 인해 전체적인 성능이 저하된다. 따라서 버스 인터커넥트는 확장성이 좋지 않다.

### 2) Switch Interconnect
스위치를 이용할 경우 다른 경로를 통해 통신할 경우 Bus와는 다르게 경합이 일어나지 않는다.   
이 방식도 직접(Direct) 연결과 간접(Indirect) 연결이 있다.

#### ※ Inter-connection의 성능을 측정하는 지표들
##### ① Node degree
한 노드에 연결된 에지(링크)의 수를 의미하며, 즉 한 번의 홉(hop)으로 도달할 수 있는 인접 노드의 수를 나타낸다

##### ② Diameter 
임의의 두 노드 사이의 가장 짧은 경로 중 가장 긴 경로를 의미한다
이는 한 프로세서에서 다른 프로세서로 메시지를 전송할 때 발생하는 최대 지연 시간을 측정하는 척도가 된다

##### ③ Bisection width
네트워크를 동일한 크기의 두 영역으로 나누기 위해 절단해야 하는 가장 적은 링크의 수이다
이는 네트워크의 이등분을 가로지르는 최대 통신 대역폭을 나타내는 지표이다.

#### ※ 데이터 전송 지표
##### ① Latency
데이터 전송 시 소스(Source)가 전송을 시작한 시점부터 목적지(Destination)에서 첫 번째 바이트를 수신하기 시작할 때까지 걸리는 시간이다

##### ② Bandwidth
링크가 데이터를 전송할 수 있는 속도로, 보통 초당 메가비트(Mb/s)나 메가바이트(MB/s) 단위로 표현된다.
목적지에서 첫 바이트를 받은 이후 데이터를 수신하는 속도를 의미하기도 한다

##### ③ Bisection Bandwidth
네트워크를 두 개의 동일한 반으로 나누는 가장 작은 절단면을 가로지르는 대역폭으로, 전체적인 네트워크의 품질을 측정하는 척도이다

#### a. Direct
직접 연결에는 아래와 같은 방식이 있다.
##### a. Linear
##### b. Ring
##### c. 2D Mesh
##### d. Torus
##### e. Fully connected
##### f. Hypercude

#### b. Indirect
간접 연결에는 아래와 같은 방식이 있다.
##### a. crossbar switch
##### b. Omega switching system

> ※ 추가 업데이트 예정
{: .prompt-tip }


# 참고자료
- 서강대학교 박성용 교수님 강의자료 - 병렬 분산 컴퓨팅  
