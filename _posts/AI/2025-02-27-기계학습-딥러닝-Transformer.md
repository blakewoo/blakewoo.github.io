---
title: 기계학습 - Deep learning - Transformer
author: blakewoo
date: 2025-8-23 22:00:00 +0900
categories: [Machine Learning]
tags: [AI, Machine Learning, Deep learning, Attention, Transformer]
render_with_liquid: false
use_math: true
---

# Transformer and "Attention Is All You Need"
## 1. 개요
기본적으로 언어라는 것은 순서가 있다. 각각의 단어들이 위치와 앞뒤 단어와의 연관성에 따라 관계를 가지고
뜻을 형성한다. 이는 이전에 포스팅했던적 있는 LSTM과 GRU가 등장하게 된 계기다.   
기존 딥러닝 모델에 비해서 순차 데이터 처리에는 LSTM과 GRU가 탁월했지만 이 역시 길이가 너무 길어지면 정보 소실 현상이 일어난다.   
흡사 사람도 말이 길어지면 논지가 불분명해질수있듯이 망각하는 현상이 일어나는 것이다.

이를 위해 Attention이라는 기술이 등장하여 이전 데이터를 다시 주의(Attention)를 일깨워주는 방식으로 정확도를 올렸지만
이 역시 조금 더 길이가 길어졌을 뿐 고질적인 문제는 해결할 수 없었다.

하지만 2017년에 구글 브레인(현재는 알파고를 만든 딥마인드사와 합병했다)에서 발표한 혁신적인 논문으로 인해
딥러닝과 LLM의 판도와 완전히 뒤바뀌는 사건이 일어났다.

간단하게 말하자면 이 "Attention Is All You Need" 논문의 핵심은 
번역 같은 순차적 작업에서 LSTM/GRU 같은 재귀(recurrent) 구조를 완전히 제거하고 자기-어텐션(self-attention)
만으로 인코더-디코더를 구성해 더 빠르게 병렬 처리할 수 있다는 내용이다.

위에서 주장하는 내용을 바탕으로 층으로 이루어진 구조가 현재의 LLM과 언어모델의 기본이 되었는데
이를 Transformer라고 한다. 이번 포스팅에서는 Transformer에 대해서 알아보겠다.

## 2. 구조
### 1) 어텐션(Attention)
### 2) Multi-Head Attention
### 3) 위치 정보 추가
### 4) Transformer

> 추가 업데이트 예정
{: .prompt-tip }

# 참고 자료
- “Attention Is All You Need” (Vaswani et al., 2017)
