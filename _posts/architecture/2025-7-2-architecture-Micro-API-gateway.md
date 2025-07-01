---
title: Architecture - Micro Service Architecture - API Gateway pattern
author: blakewoo
date: 2025-7-2 09:00:00 +0900
categories: [Service Architecture]
tags: [Service Architecture, Micro Service Architecture, API Gateway pattern] 
render_with_liquid: false
---

# Micro Service Architecture - API Gateway pattern
마이크로서비스 아키텍처에서 클라이언트와 백엔드 서비스 사이의 단일 진입점(single entry point)을 제공하는 패턴이다.
(약간 이 부분은 공부하면서 그냥 Reverse proxy로 구성하면 API Gateway pattern이 아닌가 하는 생각은 들었다.
GPT 말로는 API Gateway pattern가 Reverse proxy를 포함하는 개념이라고 하긴 했는데 좀 더 확실한 출처는 찾아보겠다)

## 1. 정의
API 게이트웨이는 모든 클라이언트 요청을 받아 적절한 마이크로서비스로 라우팅하거나,
여러 서비스에 병렬 요청을 수행해 응답을 조합해 반환하는 프록시 계층이다.

기본적으로 단일 진입점을 가지고 있는데 이는
클라이언트는 개별 서비스 주소나 포트를 알 필요 없이 게이트웨이 하나만 호출하면 되게 해준다.

크게 두 가지 종류로 나눌 수 있다.

- 프록시/라우팅: 단순히 해당 서비스로 요청을 전달
- 파닝아웃(Fan‑out): 여러 서비스에 병렬 호출 후 응답을 조합해 단일 응답 생성

## 2. 주요 역할
### 1) 라우팅(Routing)
URL 경로·헤더·파라미터 등을 기반으로 적절한 서비스로 요청 전달

### 2) API 조합(API Composition)
한 번의 클라이언트 호출로 여러 서비스의 데이터를 합쳐 응답

### 3) 보안(Security)
인증(Authentication)·인가(Authorization)를 중앙화하여 개별 서비스 부담 경감.

### 4) 부하 분산(Load Balancing) & 장애 격리(Fault Isolation)
서비스 인스턴스 풀에 트래픽 분산, 서킷 브레이커(Circuit Breaker) 적용 가능

### 5) 로깅·모니터링(Observability)
요청·응답 메트릭, 에러 로깅을 집중 수집

## 3. 장점
### 1) 클라이언트 단순화
각 서비스 주소를 몰라도 되고, 호출 로직이 경량화됨

### 2) API 최적화
모바일·웹·서드파티 등 클라이언트별로 최적화된 API 인터페이스 제공(Backends for Frontends)

### 3) 교차 관심사(Cross‑cutting Concern) 처리
보안, 로깅, 버전 관리, 요청 제한(Rate Limiting) 등을 중앙에서 관리

### 4) 버전 관리(Versioning) 지원
API 버전 변경 시 게이트웨이만 수정해 레거시·신규 버전을 동시에 운영 가능

## 4. 단점
### 1) 추가적 네트워크 홉
요청이 게이트웨이를 거치며 지연(latency)이 소폭 증가, 이는 트래픽 발생으로 추가적인 비용 발생으로 이어지기도 함.

### 2) 복잡도 증가
게이트웨이 자체가 또 하나의 마이크로서비스로 개발·배포·운영·모니터링해야 함

### 3) 단일 장애점(SPOF)
게이트웨이 장애 시 전체 시스템 접근이 불가능해지므로, 고가용성(HA) 구성 필요

> ※ 추가 업데이트 및 검증 예정이다.
{: .prompt-tip }

# 참고문헌
- [Microservices.io - api gateway pattern](https://microservices.io/patterns/apigateway.html)
- [AWS - API 게이트웨이 패턴](https://docs.aws.amazon.com/ko_kr/prescriptive-guidance/latest/modernization-integrating-microservices/api-gateway-pattern.html)
- [API 게이트웨이 패턴과 직접 클라이언트-마이크로 서비스 통신 비교](https://learn.microsoft.com/ko-kr/dotnet/architecture/microservices/architect-microservice-container-applications/direct-client-to-microservice-communication-versus-the-api-gateway-pattern)
