---
title: VectorDB - 테스트용 Dataset
author: blakewoo
date: 2025-4-28 18:00:00 +0900
categories: [Trial and error]
tags: [VectorDB, Benchmark] 
render_with_liquid: false
use_math: true
---

# VectorDB - 테스트용 Dataset
## 1. 개요
어떤 것의 성능을 알아보려면 가장 좋은건 테스트를 해보는 것이다.
물론 이론적인 것이 먼저 들어오고 이후에 실험으로 그를 뒷받침하는 형태가 가장 일반적이지만,
일단은 대부분 테스트가 무조건 들어온다는 점에서는 이견이 없을거라 생각한다.

이번 시간에는 VectorDB를 테스트해보기 위한 Dataset들에 대해 알아보겠다.

## 2. Vector 데이터 구조
크게는 두 가지로 많이들 사용되는 것 같다.   
fvecs와 bvecs이다.

<table>
    <tr>
        <td>항목</td>
        <td>fvecs 포맷</td>
        <td>bvecs 포맷</td>
    </tr>
    <tr>
        <td>데이터 타입</td>
        <td>32비트 부동소수점 (float32)</td>
        <td>8비트 부호 없는 정수 (uint8)</td>
    </tr>
    <tr>
        <td>정밀도</td>
        <td>높음 (소수점 포함)</td>
        <td>낮음 (0~255 범위의 정수)</td>
    </tr>
    <tr>
        <td>파일 크기</td>
        <td>상대적으로 큼</td>
        <td>상대적으로 작음</td>
    </tr>
    <tr>
        <td>주요 용도</td>
        <td>정확한 거리 계산이 필요한 경우</td>
        <td>메모리 효율이 중요한 경우</td>
    </tr>
    <tr>
        <td>사용 예시</td>
        <td>SIFT, GIST, Deep1M 등 고정밀 벡터</td>
        <td>SIFT1B, GIST1M 등 대규모 압축 벡터</td>
    </tr>
</table>

## 3. Vector Dataset 종류

### 1) Prep
#### a. 출처 및 용도
실제 스폰서드 광고 코퍼스에서 광고별 지역 필터 정보를 고려한 필터링 ANNS 평가를 위해 사용

#### b. 벡터 특성
64차원(float32) 텍스트 임베딩 - fvecs 타입

#### c. 규모
1,000,000개 포인트, 10,000개 쿼리

#### d. 필터 정보
광고당 평균 8.84개의 지역 필터, 전체 47개 고유 필터

### 2) DANN
#### a. 출처 및 용도
동일한 광고 코퍼스에서 지역별 서빙 시나리오 평가용으로 사용

#### b. 벡터 특성
64차원(float32) 텍스트 임베딩

#### c. 규모
3,305,317개 포인트, 32,926개 쿼리

#### d. 필터 정보
광고당 평균 3.91개의 지역 필터, 전체 47개 고유 필터

### 3) SIFT
#### a. 출처
IRISA Corpus-Texmex의 SIFT1M, fvecs 포맷

#### b. 벡터 특성
128차원(float32) SIFT 로컬 디스크립터

#### c. 규모
1,000,000개 베이스 벡터(‘database’), 10,000개 쿼리 벡터(‘test’)

#### d. 라벨(필터) 설정
ANN 벤치마크용으로 무작위 할당된 12개 필터, 포인트당 1개 평균

### 4) GIST
#### a. 출처
IRISA Corpus-Texmex의 GIST1M, fvecs 포맷

#### b. 벡터 특성
960차원(float32) GIST 전역 디스크립터

#### c. 규모
1,000,000개 베이스 벡터, 1,000개 쿼리 벡터

#### d. 라벨(필터) 설정
ANN 벤치마크용으로 무작위 할당된 12개 필터, 포인트당 1개 평균

### 5) Turing
#### a. 출처
Microsoft Turing 팀이 NeurIPS 2021 대회용으로 공개한 대규모 ANN 벤치마크 데이터셋
Bing 검색 쿼리를 Turing AGI v5 인코더로 처리하였음.

#### b. 벡터 특성
100차원 float32

#### c. 규모
벡터로 표현한 10억 개의 벡터



> ※ 추가 업데이트 및 검증 예정이고, 아직 완벽하지 않으니 참고만 바란다.
{: .prompt-tip }


# 참고문헌
- Jayaram Subramanya, Suhas, Fnu, Devvrit, Harsha Vardhan, Simhadri, Ravishankar, Krishnawamy, and Rohan, Kadekodi. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." . In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2019.
- Aditi Singh, , Suhas Jayaram Subramanya, Ravishankar Krishnaswamy, and Harsha Vardhan Simhadri. "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search." (2021).
- Siddharth Gollapudi, Neel Karia, Varun Sivashankar, Ravishankar Krishnaswamy, Nikit Begwani, Swapnil Raz, Yiyong Lin, Yin Zhang, Neelam Mahapatro, Premkumar Srinivasan, Amit Singh, and Harsha Vardhan Simhadri. 2023. Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters. In Proceedings of the ACM Web Conference 2023 (WWW '23). Association for Computing Machinery, New York, NY, USA, 3406–3416. https://doi.org/10.1145/3543507.3583552
- [텐서플로우 - sift1m](https://www.tensorflow.org/datasets/catalog/sift1m?utm_source=chatgpt.com&hl=ko)
- [Learning2hash.github.io - Microsoft Turing-ANNS-1B](https://learning2hash.github.io/publications/microsoftturinganns1B/?utm_source=chatgpt.com)
