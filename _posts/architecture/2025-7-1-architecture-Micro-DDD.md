---
title: Architecture - Micro Service Architecture - 도메인 주도 설계(DDD)
author: blakewoo
date: 2025-7-1 13:00:00 +0900
categories: [Service Architecture]
tags: [Service Architecture, Micro Service Architecture] 
render_with_liquid: false
---

# Micro Service Architecture - 도메인 주도 설계(Domain-Driven Design)

## 1. 개요
도메인 주도 설계(DDD)는 소프트웨어를 실제 비즈니스 도메인(문제 영역)의 모델과 최대한 일치시키려는 설계 접근법이다.

### 1) 핵심 아이디어   
기술적 구현이 아니라 “도메인 전문가”가 사용하는 용어와 프로세스를 코드 구조(클래스, 메서드, 변수)에 반영하여,
소프트웨어 모델과 비즈니스 모델을 동기화한다.


### 2) 바운디드 컨텍스트(Bounded Context)
큰 시스템을 의미적 경계가 분명한 여러 모델로 나누어, 각자의 언어(Ubiquitous Language)를
통해 독립적으로 진화시킨다

## 2. 장점
### 1) 비즈니스와 코드의 정렬(Alignment)   
도메인 전문가와 개발자가 동일한 용어와 개념으로 소통하므로, 요구사항 오해를 줄이고 협업 효율을 높인다.



### 2) 높은 유지보수성(Maintainability)   
모델이 잘게 분리되어 있어, 특정 도메인 로직 변경 시 관련 부분만 수정·테스트하면 되어 코드 복잡도가 낮아진다.



### 3) 모듈화된 아키텍처(Modularity)   
바운디드 컨텍스트별로 서비스나 컴포넌트를 구분할 수 있어, 독립 배포·확장 및 장애 격리가 용이하다.


### 4) 유연한 진화(Flexibility)      
요구사항이 변화해도 각 컨텍스트 내부 모델만 조정하면 되므로, 시스템 전체를 재설계할 필요가 줄어든다.


## 3. 단점
### 1) 학습 곡선(Learning Curve)   
DDD의 전략적·전술적 패턴(바운디드 컨텍스트, 애그리게잇, 리포지토리 등)을 숙지하고 적용하는 데 시간이 필요하다.


### 2) 초기 설계 복잡도(Design Overhead)   
작은 프로젝트나 단순 도메인에는 과도한 설계가 될 수 있으며, 요구사항이 불명확할 때 모델이 자주 변경되어 오히려 비효율적일 수 있다.


### 3) 인프라·조직적 비용(Infrastructure & Organizational Cost)   
각 컨텍스트별 독립 배포·테스트 환경 구축, 팀 간 경계 관리, 서비스 간 통합 메커니즘 설계 등에 추가 리소스가 필요하다.
  

## 4. 방법
### 1) 유비쿼터스 랭귀지(Ubiquitous Language) 구축   
도메인 전문가와 개발자가 공유할 용어집을 정의하고, 코드·문서·테스트 전반에 일관되게 사용하다.


### 2) 바운디드 컨텍스트(Bounded Context) 식별   
비즈니스 기능별로 모델 경계를 나누고, 각 컨텍스트마다 독립된 팀 소유·데이터 저장소·API 계약을 설계한다.


### 3) 전략적 설계(Strategic Design)   
도메인 분할(Core/Supporting/Generic Subdomain) 및 컨텍스트 매핑(Shared Kernel, ACL, Customer–Supplier 등) 패턴을 통해 서비스 간 통합 방식을 결정한다.


### 4) 전술적 설계(Tactical Design)   
엔티티(Entity), 값 객체(Value Object), 애그리게잇(Aggregate), 리포지토리(Repository), 도메인 서비스(Domain Service), 도메인 이벤트(Domain Event) 등의 구성 요소를 정의하여 모델을 구체화한다.


### 5) 지속적 리팩토링(Continuous Refactoring)   
도메인이 진화함에 따라 모델과 경계를 지속적으로 조정하며, 코드와 문서에 반영한다.


### 6) 테스트 주도 개발(Test‑Driven Development)   
단위 테스트·계약 테스트·통합 테스트를 통해 모델의 일관성을 검증하고, 변경 시 회귀를 방지합니다.    
(TDD에 대해서는 추가 포스팅이 있을 예정이다.)

> ※ 추가 업데이트 및 검증 예정이다.
{: .prompt-tip }

# 참고문헌
- [Azure - Using domain analysis to model microservices](https://learn.microsoft.com/en-us/azure/architecture/microservices/model/domain-analysis)
- [Wikipedia - Domain-driven design](https://en.wikipedia.org/wiki/Domain-driven_design)
