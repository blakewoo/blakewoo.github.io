---
title: 기계학습 - Deep learning - Fine tuning
author: blakewoo
date: 2025-8-27 21:00:00 +0900
categories: [Machine Learning]
tags: [AI, Machine Learning, Deep learning, Fine tuning]
render_with_liquid: false
use_math: true
---

# Fine tuning
## 1. 개요
사전학습(대규모 일반 데이터)된 LLM의 가중치(또는 일부 구성 요소)를 특정 작업·도메인·스타일에 맞게 추가로 학습시키는 과정이다.
목적은 같은 모델로 더 정확하고 안정적인 결과를 얻거나, 사용 사례(챗봇, 분류, 요약 등)에 특화시키는 것이다.
이미 만들어져있는것을 세부적으로(fine) 튜닝(tuning)하는 것이다. 이 방법에는 크게 두 가지 종류가 있다.

## 2. 전체 미세조정(Full fine-tuning)
거대 언어 모델의 전체 파라미터를 해당 job에 맞게 재 학습시키는 것을 말한다.    
많은 시간 및 많은 컴퓨티 파워 등 자원이 필요하다. 작은 모델을 학습시킬때나 또는 모델을 완전히 재특화해야 할 때
사용하면 효과적이다. 사실상 첫 학습과 다른점은 초기값과 학습 데이터셋이 다르다는 점만 있으며
전체적인 알고리즘은 첫 학습과 동일하다.   
때문에 모델의 모든 파라미터를 재 학습하는 것과 동일하므로 많은 연산이 필요하다.

## 3. 매개변수의 효율적 미세 조정(PEFT, Parameter-Efficient Fine-Tuning)
전체 파라미터를 첫 학습때와 같이 모두 조정한다면 매우 많은 계산량을 요한다. 특히, 점점 대형화 되어가는
대형 언어 모델 같은 경우라면 더욱 그렇다. 기본 수억개에서 수십억개의 매개변수를 가지고 있기 때문이다.    

이를 해결하기 위해 매개변수의 효율적 미세 조정(PEFT)는 기본적으로 업데이트를 하기 위한 매개변수의 수를 줄이는
모든 방법을 뜻하며 이후 이어서 서술할 여러가지 방법이 있다.

전체 모델이 아닌 일부 모델만 바꾸거나 혹은 추가적이고 작은 무언가를 붙여서 해당 부분만 변경하기 때문에
연산량이 전체 변경에 비해 압도적으로 작고 때문에 필요한 연산 리소스 및 메모리가 적다.   
이러한 PEFT 방식은 자연어 처리 사례에서는 전체 조정보다 더 안정적이라고 한다.

### 1) 부분 미세 조정(Partial fine-tuning)
- 정의/예시   
  전체 모델을 고정하고 마지막 출력층(head-only), 혹은 마지막 N개 레이어만 훈련시키는 방식.
  또는 layer norm의 bias만 조정하거나 embedding(토큰 임베딩)만 업데이트하는 방식도 포함.


- 장점   
  구현·디버깅 간단, 저장 공간 절약(모델 가중치는 고정), 빠른 실험 가능.


- 단점   
  모델 표현력을 충분히 활용하지 못해 복잡한 도메인 적응에는 한계가 있을 수 있음.


- 권장 상황   
  아주 작은 변화(특정 태스크의 출력층 재학습 등)만 필요하거나, 리소스/시간이 매우 제한된 경우

### 2) 가산 미세 조정(Additive fine-tuning)
모델 가중치는 고정하고, 작은 모듈(어댑터) 또는 학습 가능한 입력(soft-prompt / prefix) 등을 추가해 그 부분만 학습하는 방식으로,
가장 널리 쓰이는 PEFT 범주다.

#### a. 어댑터(Adapters)
- 아이디어    
  Transformer 블록 사이(또는 내부)에 작은 MLP(예: down-projection(높은 차원에서 낮은 차원으로) → nonlinearity(비선형 처리, ReLU 같은 함수)
  → up-projection(낮은 차원에서 높은 차원으로)) 형태의 모듈을 삽입하고 그 모듈만 학습. [원 논문](https://arxiv.org/pdf/1902.00751) 은
  GLUE 등 여러 태스크에서 적은 추가 파라미터로 full fine-tuning에 근접하는 성능을 보였음을 보고했다. 어댑터 방식은 태스크별로 작은 어댑터
  파일만 저장하면 되므로 멀티태스크·버전관리 측면에서 유리하다.


- 장점   
  각 태스크별로 아주 적은 추가 파라미터(수% 수준)만 필요, 원모델은 고정되어 공유 가능.


- 주의   
  어댑터 설계(크기, 삽입 위치, 활성화 함수 등)에 따라 성능 차이가 있음. 최신 라이브러리(예: Hugging Face PEFT)는
  다양한 어댑터 변형을 지원한다.

#### b. Prefix / Prompt 기반(Additive 형태)
- Prefix-Tuning    
  입력 앞에 연속적인 벡터(“virtual tokens”)를 붙이고 이 벡터만 학습. 생성형 태스크에서 특히 효과적이라는 보고가 있다(예: GPT-2, BART 등).


- Prompt-Tuning (Soft prompts)    
  discrete 텍스트 프롬프트 대신 학습 가능한 연속 벡터(soft prompt)로 모델을 condition. 모델 파라미터는 고정되고 프롬프트 벡터만 학습된다.
  대형 모델에서는 prompt-tuning이 모델 튜닝 성능에 근접할 수 있다는 관찰이 있다.


- 장점    
  매우 적은 파라미터(예: 0.1% 이하)로도 동작, 여러 태스크에 대해 prompt 벡터만 교체하면 됨.


- 단점   
  일부 작업에서 성능 격차가 있을 수 있고(특히 작은 모델/복잡한 구조에서), 프롬프트 길이·초기화 방식 등 하이퍼파라미터에 민감함.

### 3) 재매개변수화(Reparameterization)

#### a. LoRA (Low-Rank Adaptation)
- 핵심 아이디어   
  기존 가중치 행렬을 직접 변경하지 않고, attention 등 특정 행렬에 저랭크(low-rank) 형태의 보정(ΔW = A·B)만 학습. 원 모델은 freeze하고,
  두 개의 작고 저랭크 행렬(A,B)만 학습하므로 학습 파라미터가 대폭 줄어든다. LoRA는 inference 시에 원본 가중치에 병합(merge)할 수 있어
  추가 추론 지연이 거의 없다는 장점도 있다.


- 실무적 장점   
  대형 모델(수십억 파라미터)에서 유의미한 튜닝을 비교적 저자원으로 수행 가능. 다양한 프레임워크(예: Hugging Face PEFT)에서 LoRA 지원이 활발하다.

#### b. QLoRA (Quantized LoRA)
- 핵심 아이디어   
  원본 거대 모델을 4-bit(또는 저비트)로 양자화하여 GPU 메모리를 크게 절약한 상태에서, frozen quantized model 위에 LoRA 어댑터를 학습한다.
  이렇게 하면 65B급 모델도 단일 48GB GPU에서 튜닝 가능하다는 실험 결과가 보고되었다. QLoRA는 양자화된 모델을 역전파(gradient)할 때의
  수치 안정성 문제를 해결하는 여러 기법을 결합한다.


- 실무적 장점   
  큰 모델을 비교적 저비용 장비에서 튜닝할 수 있어, 연구·프로토타이핑 접근성을 크게 높임. 다만 양자화·수치안정성·퍼포먼스 트레이드오프를 이해하고
  적용해야 함.

#### ※ Adapter VS LoRA
간단히 말해서 Transformer에 입력으로 들어가는 행렬에 추가 행렬이 붙는 형태가 LoRA이고, Transformer 블록 내부에 새로운 작은 MLP 모듈을 추가하는
형태가 Adapter이다. 

> 문맥 및 내용 다듬기 및 내용 검증 예정
{: .prompt-tip }


# 참고 자료
- [LLM: 미세 조정, 정제, 프롬프트 엔지니어링](https://developers.google.com/machine-learning/crash-course/llm/tuning)
- [Hugging face - Fine-tuning](https://huggingface.co/docs/transformers/en/training)
- [IBM - What is fine tuning](https://www.ibm.com/think/topics/fine-tuning)
- Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, & Weizhu Chen. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, & Luke Zettlemoyer. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
- Neil Houlsby, , Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. "Parameter-Efficient Transfer Learning for NLP." (2019).
- Xiang Lisa Li, , and Percy Liang. "Prefix-Tuning: Optimizing Continuous Prompts for Generation." (2021).




