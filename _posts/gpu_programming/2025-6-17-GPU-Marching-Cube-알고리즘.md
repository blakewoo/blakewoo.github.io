---
title: GPU 프로그래밍 - Marching Cube 알고리즘
author: blakewoo
date: 2025-6-17 23:00:00 +0900
categories: [GPU Programming]
tags: [GPU, Marching Cube] 
render_with_liquid: false
use_math: true
---

# Marching Cube 알고리즘
## 1. 개요
CT/MRI 스캔이나 부호 거리장 데이터와 같은 3차원 스칼라장에서 등위면의 다각형 메시(일반적으로 삼각형 메시)를
추출하는 데 사용되는 컴퓨터 그래픽 알고리즘이다.

## 2. 기본적인 아이디어
격자 무늬 판이 있다고 해보자.   
어떤 모양을 Slice 하면 단면이 나온다.   
이를 격자에 대치시키면 아래와 같다.

![img.png](/assets/blog/gpu/marching_cube/img.png)

각 격자의 점을 voxel이라고 하고, 4개의 복셀로 감싸진 에지를 cell이라고 한다.
(이는 2차원일때 그런 것이고 3차원의 경우 육면체가 cell이 된다)
단면 내에 포함된 voxel가 포함되지 않은 voxel로 나뉘게 되는데 이를 기준으로 해당 Voxel이 단면의
inside냐 outside냐를 1bit로 표현할 수 있으며, 단면의 외곽이
Voxel 사이의 어느정도를 지나느냐에 따라서 각각의 iso-value를 지정할 수 있다.   
(물론 위와 같은 방식이 아닌 다른 방식도 많으나 일단은 이렇게 설명하겠다)

이 값들을 토대로 voxel 사이에 외곽이 어떻게 지나가는지 유추할 수 있으며 각 점을 연결하여
대략적인 외곽을 알아낼 수 있다.

이를 3D로 확장시키면 8개의 Voxel에서 정육면체에 어떻게 외곽이 통과하는지 알아 볼 수 있는데
정육면체의 Voxel은 총 8개이고 각 상태가 1bit씩이니 $2^{8} = 256$ 개의 경우의 수가 나온다.
여기서 대칭이나 회전으로 동일하게 만들 수 있는 중복을 제외하고 나면 고유한 형태 15개만 남는다.

![img_1.png](/assets/blog/gpu/marching_cube/img_1.png)

위 표현들을 잘 보면 알겠지만 모두 삼각형으로 표현 가능한 형태이다. 따라서 표현의 용이성과
데이터의 축소를 위해서 외곽선을 삼각형으로 연결해서 표현하는데 이를 폴리곤이라고 표현하며
일단은 한 개의 3D Cell에 대해서 표현하기 위해서는 아래와 같이 표현한다.

![img_2.png](/assets/blog/gpu/marching_cube/img_2.png)

위와 같이 Cell을 정의하고 해당 셀을 지나는 외곽을 정의할때는 Cube Index, EdgeTable과 TriTable 그리고 NumVertsTable을 사용하는데
각 내용이 어떤 정보를 포함하는지는 예시를 통해 알아보겠다.
어떤 두 개의 삼각형을 정육면체 Cell에 표시한 예시이다.

![img_3.png](/assets/blog/gpu/marching_cube/img_3.png)

v0, v1은 외곽선안에 있으니 1이고, 밖에 있는것은 0으로 표기했고
삼각형은 e1,e3,e8 과 e1,e9,e8을 이용하여 2개가 만들어져있다.   
이를 위에 언급한 형태로 표현하면 아래와 같다.
 
cubeindex = v8,v7,v6,v5....v0 = 00000011 = 3;     
EdgeTable = {e11,e10,e9 .... e0} = {0,0,1,1,0,0,0,0,1,0,1,0} = 0x30A;     
TriTable = {1,8,3,9,8,1,255,...,255};   
numVertsTable = 6;

- cubeindex는 외곽선 안에 어떤 점이 있는지 Cell단위로 표시한 값이다. v0와 v1이 inside라 각각 1을 가지며
한 Cell당 8bit로 표현가능하기 때문에 3으로 표현 할 수 있다.

- EdgeTable은 각 점이 어느 Edge를 지나는지 표현한다. 위 그림에서는 e9,e8,e3,e1을 지나기 때문에 위와 같이 표현하며
총 12bit로 표현가능하기 때문에 4bit씩 끊으면 16진수 3개로 표현 가능하다. 따라서 0x30A로 표현 가능하다.
  
- TriTable은 각 점이 삼각형으로 연결되는 집합을 표현한 것이다. 1,8,3 한세트, 9,8,1 한세트로 총 2세트이며
그외에는 쓰지 않기 때문에 255로 채워져있다.

- numVertsTable은 아까 중복 제거시 15개의 형태만 남는다고 했는데, 그 중에 몇번째에 해당하는지에 대한 내용이다.  
이전에 올린 그림과 대조해보면 6번 그림과 같음을 확인할 수 있다.


> ※ 추가 업데이트 및 검증 예정
{: .prompt-tip }


# 참고문헌
- 서강대학교 임인성 교수님 - 기초 GPU 프로그래밍 수업 자료
- Lorensen, W.E. and Cline, H.E. (1987). Marching cubes: A high resolution 3D surface construction algorithm. ACM Computer Graphics, 21(4). (SIGGRAPH ‘87)
- Claudio Montani , Riccardo Scateni and Roberto Scopigno . A modified look-up table for implicit disambiguation of Marching Cubes. The Visual Computer, 10(6), pp. 353-355.
- [NVIDIA - NVIDIA ADA GPU ARCHITECTURE](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [NVIDIA - ampere architecture White paper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
