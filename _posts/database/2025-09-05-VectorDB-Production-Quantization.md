---
title: Vector DB - Production Quantization
author: blakewoo
date: 2025-9-5 18:15:00 +0900
categories: [Database]
tags: [Database, DBMS ,VectorDB, Production Quantization]
render_with_liquid: false
use_math: true
---

# Vector DB - Production Quantization
## 1. 개요
DiskANN과 여러 VectorDB 운용간 쓰이는 양자화 방식인 Production Quantization(이하 PQ)에 예시와 함께
잘 설명해둔 한국어 포스팅이 없어서 내가 정리해서 쓰게 되었다.  
이해한대로 썼는데 혹시나 내용이 틀렸다면 메일(blakewoo0819@gmail.com)로 틀린 내용을 보내주시면 감사하겠다.

기본적으로 VectorDB는 항상 메모리가 부족하다. 이는 벡터 데이터 자체가 용량이 매우 크기 때문이다.   
끽해봐야 1,000,000개의 벡터 데이터를 메모리에 올리는데 수 GB가 필요한 경우도 있으니까 벡터 데이터가 얼마나 큰지 더 말할 것도 없다.

메모리 사용량을 줄이기 위해서 PQ를 사용하는데 메모리 사용량을 97%나 줄이고 정확도에 크게 영향을 미치지 않는 굉장한 방법이다.

## 2. 예시를 포함한 세부 절차
아래와 같이 8차원의 벡터 데이터가 6개 있다고 해보자. (혹시 차원 개념이 어렵다면, 선형대수부터 하고 오는게 좋다)   

![img.png](/assets/blog/database/vectordb/PQ/img.png)

위 데이터를 PQ하고 싶다. 여기서 정할 것은 M값인데 이 M값은 8차원을 몇 개의 작은 차원으로 나눌 것이냐를 말한다.   
이번 예시에서는 M을 4로 잡아서 2차원씩으로 쪼개보겠다.   
쪼갠다면 아래와 같이 된다.

![img_1.png](/assets/blog/database/vectordb/PQ/img_1.png)

위와 같이 2차원으로 쪼개진 값을 열(Column)단위로 살펴보면 총 4개(M개)로 나뉜다.   
여기서 K값을 정해야하는데, 위 Column 단위 값 집합을 몇 개의 군집으로 나눌 것인가에 대한 값이다.   
여기서는 편의상 K를 2로 지정하겠다.

![img_2.png](/assets/blog/database/vectordb/PQ/img_2.png)

각각의 Column에 대해서 군집화 개수를 2로 잡으면 각 Column에 있는 6개의 데이터가 3개씩 군집처리된다.   
이렇게 군집화된 3개의 데이터의 중심값을 Centroid라고 하는데 이 Centroid값이 중요하다.    

전체의 데이터를 Column당 2개(K개)를 나눌 수 있다. 이를 표로 표현하면 아래와 같다.

![img_3.png](/assets/blog/database/vectordb/PQ/img_3.png)

위와 같이 표현된걸 Codebook이라고 하며 이렇게 Codebook을 구했다면 원래 Vector와 들어올 Query에 대해서
Codebook을 통해 값을 줄일 수 있다.

![img_4.png](/assets/blog/database/vectordb/PQ/img_4.png)

위의 절차를 살펴보자. [7,2,3,0,3,9,4,6] 이라는 벡터가 있다.   
이를 M=4로 나누어서 코드북과 비교한다. 각 2차원씩을 비교하여 가까운 값의 ID를 취하면 [1,0,0,0] 과 같이 나온다.   
원본 데이터와 비교하면 사이즈가 절반이나 줄어들었다.

## 3. 평가
사실 위의 설명은 매우매우 단순화 한것으로 기본적으로 몇백 차원은 우습게 넘어가는 벡터 데이터의 경우 이 M값과 K값은
성능에 큰 영향을 미친다.

M값과 K값이 증가하면 정확도가 올라가지만 지연시간이 늘어나고 데이터는 늘어난다.   
M값과 K값이 감소하면 정확도는 내려가지만 지연시간이 줄어들고 데이터는 줄어든다.   
따라서 적당한 중간 지점을 잘 찾아서 사용하는게 매우 중요하다.

> ※ 추가 업데이트 및 검증 예정
{: .prompt-tip }

# 참고 자료
- [pinecone - Product Quantization: Compressing high-dimensional vectors by 97%](https://www.pinecone.io/learn/series/faiss/product-quantization/)
- [mildlyoverfitted - Product quantization in Faiss and from scratch](https://www.youtube.com/watch?v=PNVJvZEkuXo)
- Aditya Krishnan, , and Edo Liberty. "Projective Clustering Product Quantization." (2021).

