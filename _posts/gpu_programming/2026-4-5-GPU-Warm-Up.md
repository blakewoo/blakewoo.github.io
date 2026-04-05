---
title: GPU 프로그래밍 - Warm up
author: blakewoo
date: 2026-4-5 21:00:00 +0900
categories: [GPU Programming]
tags: [GPU, CUDA, Warm up] 
render_with_liquid: false
---

# GPU Warm up
## 1. 개요
GPU 커널을 짜서 빌드 후 돌렸을 때 성능이 훅 떨어지는 문제가 있다.   
다만, 첫번째 실행시에만 발생하거나 혹은 다른 커널을 실행시에 이런 문제가 발생하는데
이를 방지하기 위해서 Warm up이라고 하여 미리 GPU를 미리 예열하는 방식을 사용한다.

표현을 예열이라고는 했지만, 실제로 GPU의 온도를 높여놓는다는 뜻은 아니다. (온도가 올라가면 저항이 올라가기 때문에 성능이 오히려 떨어짐)
방식이 예열과 비슷해서 Warm up이라고 칭하는 것이고, 실제로는 초기화, 최적화, 클록 상승 같은 일회성 오버헤드를 미리 소모하는 방식에 가깝다.   

## 2. GPU에서 Warm up이 필요한 이유
사실 이러한 Warm up이 필요한 이유는 어째서 초기에 성능이 떨어지는 문제가 발생하냐는 이유와 동일하다.   
어째서 이러한 문제가 발생하는지는 아래의 4가지 이유를 꼽는다.

### 1) 드라이버/컨텍스트 초기화 비용
GPU와 드라이버가 아직 연결·초기화되지 않은 상태에서 애플리케이션이 GPU를 처음 건드리면 짧은 startup cost 가 발생할 수 있다.
Linux에서는 마지막 GPU 클라이언트가 종료되면 GPU가 다시 deinitialize될 수 있어서, 다음 실행의 첫 호출이 다시 느려질 수 있다.

### 2) 커널과 코드의 lazy loading
커널 코드가 종종 각 커널의 첫 launch 시점에 로드되며, 이 로딩 시간이 첫 실행에 포함될 수 있음

### 3) 프레임워크와 라이브러리의 autotuning / compilation / lazy initialization
흔히들 GPU를 사용하기 위해 사용하는 라이브러리나 프레임워크에서도 별도의 초기화나 Pre-processing을 하는데 이 과정에서 lazy-initialization이 일어난다고 한다.
이러한 lazy-initialization은 성능을 떨어뜨리는 요인이 된다.
- PyTorch의 일부 요소가 lazily initialized 됨 
- cuDNN은 새로운 입력 크기(shape) 에 대해 여러 convolution 알고리즘을 시험해 가장 빠른 것을 고를 수 있음
- torch.compile 역시 JIT 컴파일 방식이라 처음 한두 번 실행이 유의미하게 느릴 수 있고, 이후 결과를 캐시해 재사용함

### 4) 전력 관리로 인한 클록 저하
GPU 클록이 기본적으로 고정되어있지 않고 유동적이기 때문에, 유휴 시에는 낮아지고 workload가 시작되면 boost된다.
이는 GPU 전력 소모를 줄이기 위해서 적용하는 것으로 꼭 GPU에만 있는건 아니고 흔히들 쓰는 방식이다. [참고](https://blakewoo.github.io/posts/%EC%BB%B4%ED%93%A8%ED%84%B0%EA%B5%AC%EC%A1%B0-%EC%A0%80%EC%A0%84%EB%A0%A5-%EC%BB%B4%ED%93%A8%ED%84%B0-%EC%8B%9C%EC%8A%A4%ED%85%9C/)
물론 동적 클록 변화는 추론 간 성능 측정을 불안정하게 만들 수 있고 이 결과가 초기 시작시 성능이 떨어지는 요인 중 하나이다.

> 업데이트 및 추가 검증 예정
{: .prompt-tip }

# 참고문헌
- [CUDA - Lazy loading](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/lazy-loading.html)
- [pytorch - benchmark utils](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
- [nvidia - model_configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)

