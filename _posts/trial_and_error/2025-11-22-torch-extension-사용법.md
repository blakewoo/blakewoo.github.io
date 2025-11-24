---
title: Torch extension 사용법
author: blakewoo
date: 2025-11-24 23:30:00 +0900
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
$N \times N $ 정방 행렬 간의 곱셈을 하는 코드를 만들어서 구동해보도록 하겠다.

1. 디렉터리 구조 생성
아래대로 디렉터리 구조를 생성한다.
```
src/
├─ my_extension.cpp
├─ my_kernel.h
├─ my_kernel.cu
setup.py
test.py
```

실질적으로 CUDA 및 C++ 코드가 들어있는 src 폴더와 코드를 빌드하기 위한 setup.py 파일
그리고 제대로 빌드되었는지 확인하기 위한 test.py 이다.

2. CUDA 커널 생성
my_kernel.cu 라는 이름으로 파일을 만들고 커널 코드를 작성한다.
```cuda
#include <cuda_runtime.h>
#include "my_kernel.h"


__global__ void matrixMulKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}


void matrixMul(const float* A, const float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

```

2. kernel Header 생성
해당 Kernel의 함수를 인식하게 할 수 있는 커널 헤더를 만든다. 커널에 정의한 함수와 동일한 반환값과 인자를 주어야하며   
실질적으로 외부에 노출되는 함수이다.
```cpp
#ifndef MY_KERNEL_H
#define MY_KERNEL_H

#include <torch/extension.h>

void matrixMul(const float* A, const float* B, float* C, int N);

#endif // MY_KERNEL_H
```

3. CPP Wrapper 생성
커널 함수를 CPP로 감싸준다. 여기서 TORCH EXTENSION 헤더에서 제공하는 것들로 타입 체크와 그외의 필요한 바인딩을 해준다.
```cpp
#include <torch/extension.h>
#include "my_kernel.h"


void multiply_matrices(torch::Tensor a, torch::Tensor b, torch::Tensor result) {

    TORCH_CHECK(a.device() == b.device(), "Input tensors must be on the same device");
    TORCH_CHECK(a.device() == result.device(), "Result tensor must be on the same device");
    

    TORCH_CHECK(a.size(1) == b.size(0), "Incompatible dimensions for matrix multiplication");
    

    int64_t rows = a.size(0);
    matrixMul(a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), rows);
}

// Bind the functions to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multiply", &multiply_matrices, "Matrix multiplication (CUDA)");
}
```

4. Build를 위한 setup.py 생성
여기서 빌드할때 필요한 파일을 추가하며 cmdclass에서 빌드간 사용하기 위한 명령어를 정의한다.
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
  name='cuda_matrix_ops',
  ext_modules=[
    CUDAExtension(
      'cuda_matrix_ops',
      ['src/my_extension.cpp', 'src/my_kernel.cu'],
    ),
  ],
  cmdclass={
    'build_ext': BuildExtension
  }
)
```

5. 빌드
실제로 아래 명령어를 통해 빌드한다.
```shell
python3 setup.py build_ext --inplace
```

6. 테스트
실제 구동되면 Result 이후에 4x4 배열의 곱이 나온다.
```python
import sys
import torch

# 현재 디렉토리를 경로에 추가 (로컬 .so 파일 인식)
sys.path.insert(0, '{빌드한 so 파일이 있는 폴더 경로}')

import cuda_matrix_ops

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = torch.randn(4, 4, device=device, dtype=torch.float32)
    b = torch.randn(4, 4, device=device, dtype=torch.float32)
    result = torch.zeros(4, 4, device=device, dtype=torch.float32)
    
    cuda_matrix_ops.multiply(a, b, result)
    print("Result:", result)

if __name__ == "__main__":
    test()
```

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
