---
title: GPU 프로그래밍 - Warm up
author: blakewoo
date: 2026-4-7 21:00:00 +0900
categories: [GPU Programming]
tags: [GPU, CUDA, Warm up] 
render_with_liquid: false
---

# GPU Warm up
## 1. 개요
GPU 커널을 짜서 빌드 후 돌렸을 때 초기 한번에 한 해 성능이 훅 떨어지는 문제가 있다.   
완전 첫번째 실행시에만 발생하거나 혹은 완전 다른 커널을 실행시에 이런 문제가 발생하는데
이를 방지하기 위해서 Warm up이라고 하여 미리 GPU를 미리 예열하는 방식을 사용한다.

표현을 예열이라고는 했지만, 실제로 GPU의 온도를 높여놓는다는 뜻은 아니다. (온도가 올라가면 회로 저항이 올라가기 때문에 성능이 오히려 떨어짐)
방식이 예열과 비슷해서 Warm up이라고 칭하는 것이고, 실제로는 초기화, 최적화, 클록 상승 같은 일회성 오버헤드를 미리 소모하는 방식에 가깝다.   

## 2. GPU에서 Warm up이 필요한 이유 (성능 저하의 이유)
사실 이러한 Warm up이 필요한 이유는 어째서 초기에 성능이 떨어지는 문제가 발생하냐는 이유와 동일하며   
어째서 이러한 문제가 발생하는지는 아래의 이유들을 꼽는다.

### 1) 커널 로딩 / 초기화 비용 처리
커널 로딩과 초기화 비용을 처리하는데 필요하다고 한다면 대략적으로는 이해가 가지만
이러한 커널 로딩과 초기화라는게 좀 애매하다. 커널 로딩이 GPU를 사용하기 위한 프로그램을 VRAM에 Loading하는것 까지는 알겠으나 초기화란 무엇인가?

그냥 말하자면 첫 CUDA 호출 전후에 한꺼번에 붙는 준비 비용을 말하며, 이를 순서대로 잘개 쪼개보면 아래와 같다.
 
#### a. 드라이버/컨텍스트 초기화 비용
드라이버 API를 쓰면 cuInit()이 먼저 필요하고, 그다음 특정 디바이스에 연결된 CUDA context가 생성되어 현재 호스트 스레드에 붙는다.
컨텍스트는 CPU 프로세스 비슷한 개념이라서, 그 안에 주소 공간과 모듈, 스트림, 이벤트, 할당된 메모리 같은 리소스가 binding 된다.
그래서 “첫 호출이 느리다”는 건 실제 연산 전에 이 실행 환경을 세팅하는 비용이 포함하는 것이다.

#### b. 디바이스 코드 로딩
애플리케이션에 들어 있는 CUBIN/FATBIN/PTX 중 현재 디바이스에서 실행할 코드를 골라 GPU 쪽에서 쓸 수 있게 적재한다.
CUDA는 이 로딩 방식을 CUDA_MODULE_LOADING으로 제어할 수 있는데, LAZY면 첫 커널 호출이나 첫 함수 핸들 추출 시점까지 실제 커널 로딩을 미루고,
EAGER면 프로그램 초기화 시점에 모듈과 커널을 한꺼번에 올리게 된다.

#### c. PTX → 실제 GPU용 바이너리(SASS) JIT 컴파일(옵션)
필요한 경우 발생하는 것으로 런타임에 로드된 PTX는 디바이스 드라이버가 추가 컴파일하며,
이 JIT는 애플리케이션 로드 시간을 늘릴 수 있다.
대신 생성된 바이너리는 compute cache에 저장되어 다음 실행에서 같은 컴파일을 반복하지 않도록 되어 있으며
반대로 캐시가 비활성화됐거나 드라이버가 바뀌었거나, 해당 아키텍처용 바이너리가 없어서 매번 PTX를 쓰게 되면 첫 실행 비용이 다시 커질 수 .

#### d. 모듈 데이터와 심볼 관련 로딩
커널 코드만이 아니라 모듈에 딸린 데이터도 로딩 대상인데, CUDA_MODULE_DATA_LOADING=LAZY면 이런 데이터 로딩도 늦춰진다.
CUDA 문서에는 lazy data load가 컨텍스트 동기화를 요구할 수 있고, 그게 동시 실행을 느리게 만들 수 있다고 나와있다.

### 2) 프레임워크와 라이브러리의 autotuning / compilation / lazy initialization
흔히들 GPU를 사용하기 위해 사용하는 라이브러리나 프레임워크에서도 별도의 초기화나 Pre-processing을 하는데 이 과정에서 lazy-initialization이 일어난다고 한다.
이러한 lazy-initialization은 성능을 떨어뜨리는 요인이 된다.
- PyTorch의 일부 요소가 lazily initialized 됨 
- cuDNN은 새로운 입력 크기(shape) 에 대해 여러 convolution 알고리즘을 시험해 가장 빠른 것을 고를 수 있음
- torch.compile 역시 JIT 컴파일 방식이라 처음 한두 번 실행이 유의미하게 느릴 수 있고, 이후 결과를 캐시해 재사용함

### 3) 전력 관리로 인한 클럭 저하
GPU 클록이 기본적으로 고정되어있지 않고 유동적이기 때문에, 유휴 시에는 낮아지고 workload가 시작되면 boost된다.
이는 GPU 전력 소모를 줄이기 위해서 적용하는 것으로 꼭 GPU에만 있는건 아니고 흔히들 쓰는 방식이다. [참고](https://blakewoo.github.io/posts/%EC%BB%B4%ED%93%A8%ED%84%B0%EA%B5%AC%EC%A1%B0-%EC%A0%80%EC%A0%84%EB%A0%A5-%EC%BB%B4%ED%93%A8%ED%84%B0-%EC%8B%9C%EC%8A%A4%ED%85%9C/)
물론 동적 클록 변화는 추론 간 성능 측정을 불안정하게 만들 수 있고 이 결과가 초기 시작시 성능이 떨어지는 요인 중 하나이다.

> 업데이트 및 추가 검증 예정
{: .prompt-tip }

# 참고문헌
- [CUDA - Lazy loading](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/lazy-loading.html)
- [pytorch - benchmark utils](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
- [nvidia - model_configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
- [NVIDIA - CUDA Programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf)
