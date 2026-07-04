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

## 2. 기 임베딩 된 Vector 데이터셋의 구조
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

## 3. Beir Dataset
21년도에 나온 논문에 포함된 Beir Dataset이다. Vector 검색 논문에서는 매우 많이 쓰는 일반적인 데이터셋이다.   
BEIR은 다양한 정보 검색(IR) 작업을 포함하는 heterogeneous 한 벤치마크이며 또한, 벤치마크 내에서 자연어 처리(NLP) 기반 검색 모델을 평가하기 위한
공통적이고 간편한 프레임워크를 제공한다고 되어있다.

실제 데이터셋은 [github 사이트](https://github.com/beir-cellar/beir) 에서 받을 수 있으며 아예 Python 패키지로 만들어져있기도 해서
제공하는 패키지에 데이터셋 이름만 입력하면 해당 데이터셋을 받아주기까지한다.

각 데이터 셋의 이름과 간단한 설명은 아래와 같다.

<table class="html-table-editor-output">
  <thead>
    <tr>
      <th>Dataset 이름</th>
      <th>Queries 개수</th>
      <th>Corpus 개수</th>
      <th>설명</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MSMARCO</td>
      <td>6,980</td>
      <td>8.84M</td>
      <td>웹 검색 질의와 문서로 구성된 대규모 일반 도메인 retrieval 데이터셋</td>
    </tr>
    <tr>
      <td>TREC-COVID</td>
      <td>50</td>
      <td>171K</td>
      <td>COVID-19 관련 논문 검색을 위한 생의학 정보검색 데이터셋</td>
    </tr>
    <tr>
      <td>NFCorpus</td>
      <td>323</td>
      <td>3.6K</td>
      <td>영양,건강,의학 중심의 소규모 전문 도메인 retrieval 데이터셋</td>
    </tr>
    <tr>
      <td>BioASQ</td>
      <td>500</td>
      <td>14.91M</td>
      <td>생의학 질문응답과 문헌 검색을 평가하는 의료 도메인 데이터셋</td>
    </tr>
    <tr>
      <td>NQ</td>
      <td>3,452</td>
      <td>2.68M</td>
      <td>실제 사용자 질문에 가까운 Natural Questions 기반 위키피디아 검색 데이터셋</td>
    </tr>
    <tr>
      <td>HotpotQA</td>
      <td>7,405</td>
      <td>5.23M</td>
      <td>여러 문서를 함께 찾아야 답할 수 있는 multi-hop QA 검색 데이터셋</td>
    </tr>
    <tr>
      <td>FiQA-2018</td>
      <td>648</td>
      <td>57K</td>
      <td>금융 뉴스와 투자 관련 질의를 다루는 금융 도메인 검색 데이터셋</td>
    </tr>
    <tr>
      <td>Signal-1M(RT)</td>
      <td>97</td>
      <td>2.86M</td>
      <td>트윗(짧은 소셜 미디어 텍스트) 검색을 위한 데이터셋</td>
    </tr>
    <tr>
      <td>TREC-NEWS</td>
      <td>57</td>
      <td>595K</td>
      <td>뉴스 기사 검색 성능을 평가하는 뉴스 도메인 데이터셋</td>
    </tr>
    <tr>
      <td>Robust04</td>
      <td>249</td>
      <td>528K</td>
      <td>어떤 주장에 대한 반대 논거를 찾는 argument retrieval 데이터셋</td>
    </tr>
    <tr>
      <td>ArguAna</td>
      <td>1,406</td>
      <td>8.67K</td>
      <td>주장/반박 중심의 논증 검색을 평가하는 argument retrieval 데이터셋</td>
    </tr>
    <tr>
      <td>Touche-2020</td>
      <td>49</td>
      <td>382K</td>
      <td>주장/반박 중심의 논증 검색을 평가하는 argument retrieval 데이터셋</td>
    </tr>
    <tr>
      <td>CQADupstack</td>
      <td>13,145</td>
      <td>457K</td>
      <td>CQA 포럼에서 의미가 같은 중복 질문을 찾는 데이터셋</td>
    </tr>
    <tr>
      <td>Quora</td>
      <td>10,000</td>
      <td>523K</td>
      <td>Quora 질문 쌍에서 중복 질문을 판별하는 데이터셋</td>
    </tr>
    <tr>
      <td>DBPedia</td>
      <td>400</td>
      <td>4.63M</td>
      <td>위키 기반 지식베이스에서 엔티티를 검색하는 entity retrieval 데이터셋</td>
    </tr>
    <tr>
      <td>SCIDOCS</td>
      <td>1,000</td>
      <td>25K</td>
      <td>인용 관계가 중요한 과학 논문 검색 및 citation prediction 데이터셋</td>
    </tr>
    <tr>
      <td>FEVER</td>
      <td>6,666</td>
      <td>5.42M</td>
      <td>주장(claim)을 근거 문서로 검증하는 fact verification 데이터셋</td>
    </tr>
    <tr>
      <td>Climate-FEVER</td>
      <td>1,535</td>
      <td>5.42M</td>
      <td>기후 변화 관련 주장 검증을 위한 fact verification 데이터셋</td>
    </tr>
    <tr>
      <td>SciFact</td>
      <td>300</td>
      <td>5K</td>
      <td>과학 논문 근거를 바탕으로 주장 검증을 수행하는 데이터셋</td>
    </tr>
  </tbody>
</table>


> ※ 추가 업데이트 및 검증 예정이고, 아직 완벽하지 않으니 참고만 바란다.
{: .prompt-tip }


# 참고문헌
- Jayaram Subramanya, Suhas, Fnu, Devvrit, Harsha Vardhan, Simhadri, Ravishankar, Krishnawamy, and Rohan, Kadekodi. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." . In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2019.
- Aditi Singh, , Suhas Jayaram Subramanya, Ravishankar Krishnaswamy, and Harsha Vardhan Simhadri. "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search." (2021).
- Siddharth Gollapudi, Neel Karia, Varun Sivashankar, Ravishankar Krishnaswamy, Nikit Begwani, Swapnil Raz, Yiyong Lin, Yin Zhang, Neelam Mahapatro, Premkumar Srinivasan, Amit Singh, and Harsha Vardhan Simhadri. 2023. Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters. In Proceedings of the ACM Web Conference 2023 (WWW '23). Association for Computing Machinery, New York, NY, USA, 3406–3416. https://doi.org/10.1145/3543507.3583552
- [텐서플로우 - sift1m](https://www.tensorflow.org/datasets/catalog/sift1m?utm_source=chatgpt.com&hl=ko)
- [Learning2hash.github.io - Microsoft Turing-ANNS-1B](https://learning2hash.github.io/publications/microsoftturinganns1B/?utm_source=chatgpt.com)
- Nandan Thakur, , Nils Reimers, Andreas Ruckle, Abhishek Srivastava, and Iryna Gurevych. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." . In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).2021.

