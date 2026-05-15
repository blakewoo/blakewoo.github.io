---
title: GPU 프로그래밍 - CUDA Memory Management
author: blakewoo
date: 2026-5-15 22:00:00 +0900
categories: [GPU Programming]
tags: [GPU, CUDA] 
render_with_liquid: false
---

# CUDA Memory Management
## 1. CUDA 메모리 구조
CUDA 메모리는 데이터의 대역폭과 지연 시간(Latency)의 격차를 줄이기 위해 계층적 구조를 가진다.
크게 On-chip 메모리와 Off-chip 메모리로 나눈다.

![img.png](/assets/blog/gpu/cuda_memory_management/img.png)

### 1) On-chip 메모리
레지스터, 공유 메모리(Shared Memory), L1/L2 캐시 등이 포함되며 대역폭이 매우 높고 지연 시간이 짧다.

### 2) Off-chip 메모리
전역 메모리(Global Memory), 로컬 메모리(Local Memory), 상수(Constant, 읽기 전용) 및 텍스처(Texture, 읽기 전용)
메모리가 포함되며 용량은 크지만 지연 시간이 길다.

## 2. Register
On-chip 메모리이며 가장 빠르다. 그리고 각 스레드마다 독립적으로 할당되는 저장 공간이다.
주로 자동 변수(스레드가 자신만 별도로 쓰는 변수), 스칼라 변수들을 저장하며, 스레드 블록 내의 스레드 수와 스레드당 레지스터 사용량은 GPU의 점유율(Occupancy)을
결정하는 중요한 요소이다.

## 3. Local Memory
Off-chip 메모리이며 실제로는 전역 메모리와 동일한 저장 공간을 사용한다.
이름은 로컬이지만 물리적으로는 On-chip이 아니기 때문에 전역 메모리만큼 느리다.
레지스터 공간이 부족하여 변수가 밀려날 때(Spilling) 또는 스레드 전용의 큰 배열을 저장할 때 사용되며, 데이터의 유효 범위(Scope)는 해당 스레드로 한정된다.

## 4. Shared Memory
On-chip 메모리이며 스레드 블록 내의 모든 스레드가 공유할 수 있는 빠르고 명시적인 관리형 캐시이다.
전역 메모리보다 훨씬 빠르며 자주 접근하는 데이터의 복사본을 저장하거나, 같은 블록 내의 스레드들끼리 데이터를 교환할 때 사용한다.
기본적으로 L1 캐시를 같이 공유하는데, Shared Memory로 잡는 만큼 L1 캐시의 크기가 줄어든다.

일반적으로 아래의 용도로 많이 사용된다.   

- 전역 메모리에 접근해야 하는 매우 자주 사용되는 데이터(예: 카운터 배열)를 저장하는 장소.
- 전역 메모리(프로그램 관리 캐시)에 저장될 데이터를 여러 번 접근해야 하는 경우, 해당 데이터의 빠른 미러링(예: 입력 데이터의 일부).
- SM 내 코어들이 데이터를 빠르게 공유하는 방법.

## 5. Global Memory
Off-chip 메모리이며 흔히들 GPU DRAM이나 혹은 HBM인 경우도 있다.
가장 크고 지연 시간이 길며, 모든 그리드와 스레드가 접근할 수 있는 가장 일반적인 메모리로 대부분 VRAM이라고 하면 이 Global Memory를 말한다.
cudaMalloc과 cudaFree를 통해 동적으로 할당 및 해제하며, 데이터 접근은 L2 캐시를 거쳐 이루어진다.

## 6. Zero-copy Memory (Mapped Memory)
원래라면 GPU에서 연산을 할때 HOST MEMORY에서 GPU RAM을 데이터를 복사해와서 연산해야하지만, Zero-copy Memory 기술을 사용하면
Host의 메모리의 Pinned Memory를 GPU가 직접 참조 할 수 있게 매핑된다. (CPU 메모리를 GPU 메모리 주소로 Mapping 해주기 때문)

![img.png](/assets/blog/gpu/cuda_memory_management/img_1.png)

명시적인 데이터 복사라던지 GPU RAM이 부족할때 Host memory를 가져다 쓸수있는 장점이 있지만, 결국에는 PCIe 버스를 통해
데이터를 가져오게 되므로 속도는 여전히 느리며, 사실상 개발자의 편의성을 위해서 있는 기술이라고 생각하면 된다.

## 7. Bank conflict
공유 메모리는 보통 32개의 뱅크(Bank)로 나뉘어 관리된다. 그런데 한 워프(32스레드) 내의 여러 스레드가
동일한 뱅크의 서로 다른 주소에 동시에 접근하려 할 때 발생하는 문제가 바로 Bank conflict이다.
접근 요청이 직렬화(Serialization)되어 메모리 성능이 급격히 저하되는 것이다.

### ※ Bank와 align의 관계
흔히들 CPU를 이용해서 코딩할 때도 구조체를 잡으면 큰 것 기준으로 align이 맞춰지게 된다.   
가령 int, char, int가 혼재된 방식으로 구조체를 선언하면 align으로 인해 크기가 12bytes가 되는데
이러한 이유는 host memory의 bank 구조로 인해 발생한다.

### 1) 해결 방식 1 - Padding 추가
패딩(Padding)은 데이터 배열의 행(Row) 사이에 의미 없는 더미 요소(extra elements)를
추가하여 메모리 접근 간격(Stride)을 변경하는 방식이다.

### 2) 해결 방식 2 - Re-order
리오더(Reorder)는 데이터의 물리적 구조를 바꾸는 대신,
스레드가 메모리에 접근하는 순서나 인덱스 매핑 방식을 변경하여 컨플릭트를 피하는 방식이다.

### 3) 해결 방식 3 - BroadCasting
브로드캐스트는 뱅크 컨플릭트의 예외 상황으로, 워프 내의 모든 스레드가 정확히 동일한 주소(동일한 32비트 워드 내)를 읽으려고 할 때 발생한다.
하드웨어가 해당 요청을 감지하면, 요청된 주소의 데이터를 워프 내의 모든 스레드에 동시에 전달한다.
이 작업은 단일 브로드캐스트 트랜잭션으로 처리되기 때문에 뱅크 컨플릭트가 발생하지 않으며, 매우 효율적이고 빠르다

## 8. Memory Coalesced Access
워프 내의 연속된 스레드들이 연속적인 전역 메모리 주소에 접근할 때, 이를 하나의 메모리 트랜잭션으로 합쳐서 처리하는 최적화 기법이다.
불필요한 메모리 트랜잭션 횟수를 최소화하여 전역 메모리 대역폭을 극대화한다.
반대로 스레드들이 흩어진 주소에 접근(Non-coalesced)하면 최대 32번의 트랜잭션이 발생할 수 있다.

## 9. Unified Virtual Address
CPU와 모든 GPU가 하나의 통합된 가상 주소 공간을 공유하는 기술이다.
포인터 값이 시스템 전체에서 고유해지므로, 런타임이 포인터만 보고 데이터가 호스트에 있는지
어느 GPU에 있는지 자동으로 판단할 수 있다.

## 10. Unified Memory
CPU와 GPU가 동일한 포인터를 사용하여 접근할 수 있는 관리형 공유 메모리이다.
cudaMallocManaged로 할당하며, 시스템이 데이터가 필요한 시점에 호스트와 디바이스 사이에서 메모리 페이지를 자동으로 이동(On-demand migration)시킨다.
개발자가 명시적으로 cudaMemcpy를 호출할 필요가 없어 프로그래밍 모델이 매우 단순해지며 효율적인 메모리 활용이 가능하다.

> ※ 추가 업데이트 및 검증 예정이다.
{: .prompt-tip }

# 참고자료
- 서강대학교 박성용 교수님 강의자료 - 병렬 분산 컴퓨팅

# 원문 참고자료들
- Peter S. Pacheco, An Introduction to Parallel Programming,  Elsevier Inc. (Morgan Kaufmann), 2011, ISBN 978-0-12-374260-5
- Gerassimos Barlas, Multicore and GPU Programming – An Integrated Approach, Elsevier Inc. (Morgan Kaufmann), 2015, ISBN 978-0-12-417137-4.
