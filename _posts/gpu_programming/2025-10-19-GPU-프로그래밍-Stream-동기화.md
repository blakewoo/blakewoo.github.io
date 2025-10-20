---
title: GPU 프로그래밍 - CUDA Stream 간 동기화
author: blakewoo
date: 2025-10-20 23:00:00 +0900
categories: [GPU Programming]
tags: [GPU, CUDA, Synchronization] 
render_with_liquid: false
---

# CUDA Stream 간 동기화
이전에 cuda 문법에 대해서 포스팅한 적이 있다.   
이번에는 Stream 동기화에 대해서 좀 더 알아보도록 하겠다.

## 1. 개요
GPU를 잘 사용한다는 것은 GPU를 100% 활용한다는 것이다.   
일반적으로 Branch가 적고 Memory coalescing을 지키며 Warp Occupancy가 높고,
Shared Memory를 적극적으로 활용하면 일반적으로는 GPU를 잘 쓴다고 할 수 있다.
(물론 더 잘하는 사람들은 Texture Memory까지 사용하여 가속화하기도한다)

있는 메모리를 잘 쓰는 것도 매우 중요하지만, 여기에 더불어 중요한게 있으니 Synchronization을 이용한
Computation - Communication Overlap이다.

기본적으로 GPU는 아래의 순서대로 작업한다.

1. HOST에서 GPU의 VRAM으로 데이터를 복사해온다.
2. VRAM에서 해당 데이터를 주어진 커널에 따라 연산한다.
3. 연산한 결과를 VRAM에서 HOST RAM으로 복사한다.

이 과정이 순차적으로 1이 끝나고 2가 시작되고, 2가 끝나고 3이 시작되는 식으로 처리된다면
일반적인 프로그램 실행 순서겠지만, 사실 1,3은 생각보다 꽤 긴 시간이 필요하다.

일반적인 Memory에 접근하는 것으로 생각할 수 있지만, PCI-express를 타고 데이터가 왔닥 갔다 하기 때문이다.   
때문에 CUDA에서 제공하는 Stream 기능을 이용하여 데이터 전송 부분을 비동기적으로 운용하는 것이다.

## 2. Stream Synchronization
STREAM을 한 개만 사용하여 VECTOR 두 개를 더하는 연산을 3번 한다고 해보자.
각각 다른 값을 가졌지만 동일한 크기 N을 가진 VECTOR 1~6을 1-2, 3-4, 5-6 끼리 더하는 연산이다.
VECTOR 1~6은 Pinned Memory에 저장되어있으며 코드로 나타내면 아래와 같다.

```cuda
// .... 전략 ......
cudaMalloc(&GPU_vector1,sizeof(float)*N);
cudaMalloc(&GPU_vector2,sizeof(float)*N);
cudaMalloc(&GPU_vector1_2,sizeof(float)*N);
cudaMalloc(&GPU_vector3,sizeof(float)*N);
cudaMalloc(&GPU_vector4,sizeof(float)*N);
cudaMalloc(&GPU_vector3_4,sizeof(float)*N);
cudaMalloc(&GPU_vector5,sizeof(float)*N);
cudaMalloc(&GPU_vector6,sizeof(float)*N);
cudaMalloc(&GPU_vector5_6,sizeof(float)*N);

cudaMemcpy(GPU_vector1,vector1,sizeof(float)*N,cudaMemcpyHostToDevice);
cudaMemcpy(GPU_vector2,vector2,sizeof(float)*N,cudaMemcpyHostToDevice);
cudaMemcpy(GPU_vector3,vector3,sizeof(float)*N,cudaMemcpyHostToDevice);
cudaMemcpy(GPU_vector4,vector4,sizeof(float)*N,cudaMemcpyHostToDevice);
cudaMemcpy(GPU_vector5,vector5,sizeof(float)*N,cudaMemcpyHostToDevice);
cudaMemcpy(GPU_vector6,vector6,sizeof(float)*N,cudaMemcpyHostToDevice);

dim3 threads = {512,1,1};
dim3 blocks = {(N+threads.x-1)/threads.x,1,1};

vectorAdd <<< blocks, threads >>> (GPU_vector1,GPU_vector2,GPU_vector1_2,N);
vectorAdd <<< blocks, threads >>> (GPU_vector3,GPU_vector4,GPU_vector3_4,N);
vectorAdd <<< blocks, threads >>> (GPU_vector5,GPU_vector6,GPU_vector5_6,N);

cudaMemcpy(vector1_2,GPU_vector1_2,sizeof(float)*N,cudaMemcpyDeviceToHost);
cudaMemcpy(vector3_4,GPU_vector3_4,sizeof(float)*N,cudaMemcpyDeviceToHost);
cudaMemcpy(vector5_6,GPU_vector5_6,sizeof(float)*N,cudaMemcpyDeviceToHost);
// ... 후략 ....
```

별도의 스트림을 지정하지 않았지만 별도의 스트림을 지정하지 않았다면 default stream에서 모두 실행된다.   
이를 그림으로 표현하면 아래와 같다.

![img.png](/assets/blog/gpu/synchronization/img.png)

기본적으로 cudaMalloc은 host와 묵시적으로 동기화된다. 기본 stream만 사용하면 위와 같은 그림으로 실행된다.

> ※ 내용 업데이트 및 추가적인 검증 예정
{: .prompt-tip }
> 
### 1) Double buffering
하나의 stream을 하나 더 추가하여 구동하면 아래와 같은 그림이 된다.

![img_1.png](/assets/blog/gpu/synchronization/img_1.png)

Steam 하나는 데이터 전송용으로 다른 하나는 계산용으로 사용하는 것이다.
Stream이 따로 작동하여 overlap되면 overlap 된 시간만큼 전체 시간 소요가 줄어든다.

> ※ 내용 업데이트 및 추가적인 검증 예정
{: .prompt-tip }
> 
### 2) Triple Buffering
stream을 하나 더 추가하여 하나는 출력 stream으로 사용할 경우의 그림이다.

![img_2.png](/assets/blog/gpu/synchronization/img_2.png)

overlap이 추가적으로 일어나서 좀 더 실행 시간이 감소한 것을 볼 수 있다.



> ※ 내용 업데이트 및 추가적인 검증 예정
{: .prompt-tip }

# 참고문헌
- [NVIDIA - NVIDIA ADA GPU ARCHITECTURE](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [CUDA TOOLKIT DOCUMENTATION - api-sync-behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
