---
title: Torch extension 사용법
author: blakewoo
date: 2025-11-22 23:30:00 +0900
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


### 2) Just-In-Time (JIT) 컴파일
개발 과정에서 빠른 반복을 위해 torch.utils.cpp_extension.load() 함수를 사용하여 즉석에서
컴파일할 수 있다.

1. C++ 로 CUDA 커널을 짠다.
2. C++ 로 인터페이스를 짠다.
3. Python에서 커널 바인드 및 컴파일해서 호출해서 사용한다.



> 추가 업데이트 예정
{: .prompt-tip }

# 참고문헌
- https://tutorials.pytorch.kr/advanced/cpp_custom_ops.html
