---
title: Torch extension 사용법
author: blakewoo
date: 2025-11-23 23:30:00 +0900
categories: [Trial and error]
tags: [GPU, CUDA, Torch extension] 
render_with_liquid: false
---

# Torch extension 사용법
## 1. 개요
원래 GPU를 구동할 수 있는 코드는 C++을 이용해서만 짤 수 있었다.   
하지만 개발간 Python이 매우 많이 사용됨으로 인해, C++로 짜여진 커널을 Python에서 구동할 수 있으면
좋겠다는 니즈가 생겼고 이로 인해 많은 라이브러리들이 CUDA 커널을 지원하기 시작했다.   

이번에는 많은 라이브러리들 중에 Torch extension에 대해서 이야기할 것이다.
PyTorch Extension은 C++/CUDA 커스텀 코드를 PyTorch에 통합하여 성능을
최적화하거나 새로운 연산을 추가할 수 있게 해주는 기능이다.
기본 PyTorch 연산으로 표현할 수 없거나 성능 최적화가 필요한 경우에 사용한다.

## 2. 방법
총 2가지 사용 방법이 있다.
### 1) setuptools를 이용한 Ahead-of-Time 빌드
가장 일반적인 방법으로, setup.py 파일을 작성하여 사전 컴파일한 뒤 Python에서 호출해서 사용하는 방식이다.

순서는 아래와 같다.

1. C++ 로 CUDA 커널을 짠다.
2. C++ 로 인터페이스를 짠다.
3. CUDA 커널과 인터페이스를 Bind 한다.
4. Extension을 setup tool로 컴파일 한다. 
5. 컴파일된 Extension을 Python에서 호출해서 사용한다.

#### a. 사용 예시
이번 예시에서는 행렬 곱셈을 하는 커널을 만들어서 pytorch에서 사용할 수 있도록 해보겠다.

1. CUDA 커널 생성
```cuda
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA 커널 정의
template <typename scalar_t>
__global__ void muladd_cuda_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t c,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] * b[idx] + c;
    }
}

// CUDA 함수 구현
torch::Tensor muladd_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    double c) {
    
    // 연속 메모리 보장
    auto a_contig = a.contiguous();
    auto b_contig = b.contiguous();
    
    // 출력 텐서 생성
    auto output = torch::empty_like(a);
    
    const int size = a.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    // 현재 CUDA 스트림 가져오기
    const auto stream = at::cuda::getCurrentCUDAStream();
    
    // 타입에 따라 커널 실행
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "muladd_cuda", ([&] {
        muladd_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            a_contig.data_ptr<scalar_t>(),
            b_contig.data_ptr<scalar_t>(),
            static_cast<scalar_t>(c),
            output.data_ptr<scalar_t>(),
            size
        );
    }));
    
    return output;
}

```

2. CPP Wrapper 생성
```cpp
#include <torch/extension.h>

// CUDA forward 선언
torch::Tensor muladd_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    double c);

// CPU 구현
torch::Tensor muladd_cpu(
    const torch::Tensor& a,
    const torch::Tensor& b,
    double c) {
    return a * b + c;
}

// 디바이스 분기
torch::Tensor muladd(
    const torch::Tensor& a,
    const torch::Tensor& b,
    double c) {
    
    TORCH_CHECK(a.sizes() == b.sizes(), 
                "Input tensors must have same shape");
    
    if (a.is_cuda()) {
        return muladd_cuda(a, b, c);
    }
    return muladd_cpu(a, b, c);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("muladd", &muladd, "Multiply-add operation");
}

```

3. Build를 위한 setup.py 생성
```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='MatmulExtension',
    ext_modules=[
        CUDAExtension(
            'MatmulExtension',
            ['MatmulExtension.cpp', 'MatmulExtension.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```

4. 빌드된 패키지 사용

### 2) Just-In-Time (JIT) 컴파일
개발 과정에서 빠른 반복을 위해 torch.utils.cpp_extension.load() 함수를 사용하여 즉석에서
컴파일할 수 있다.

1. C++ 로 CUDA 커널을 짠다.
2. C++ 로 인터페이스를 짠다.
3. Python에서 커널 바인드 및 컴파일해서 호출해서 사용한다.

#### b. 사용 예시
python 코드 내에서 load하거나 inline code로 불러온다.

```python
# python_usage.py (JIT 예시)
from torch.utils.cpp_extension import load_inline
import torch

# C++ 및 CUDA 소스 코드를 문자열로 정의
cuda_source = """
__global__ void add_kernel(const float* x, const float* y, float* z, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        z[index] = x[index] + y[index];
    }
}
"""

cpp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA 커널 래퍼
void add_wrapper(at::Tensor x, at::Tensor y, at::Tensor z, int N) {
    // 적절한 그리드 및 블록 크기 계산
    dim3 blocks((N + 255) / 256, 1, 1);
    dim3 threads(256, 1, 1);
    
    // 현재 CUDA 스트림 가져오기 (분산 학습 등에서 중요)
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    add_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), N);
}

// Python 바인딩
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_add", &add_wrapper, "Custom add kernel (CUDA)");
}
"""

# 확장 프로그램 컴파일 및 로드
my_cuda_extension = load_inline(
    name="MatmulExtension",
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['Matmul'],
    verbose=True
)
```

빌드가 완료되면 아래와 같이 불러와서 사용한다.

```python
import torch
# JIT 사용 시: my_cuda_extension는 이미 로드됨

# CUDA 텐서 생성
x = torch.randn(1000).cuda()
y = torch.randn(1000).cuda()
z = torch.empty_like(x).cuda()

# 사용자 정의 CUDA 함수 호출
MatmulExtension.Matmul(x, y, z)

# 결과 확인
expected = x + y
assert torch.allclose(z, expected)
```

> 추가 업데이트 예정
{: .prompt-tip }

# 참고문헌
- https://tutorials.pytorch.kr/advanced/cpp_custom_ops.html
