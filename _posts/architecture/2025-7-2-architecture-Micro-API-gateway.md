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

![img.png](/assets/blog/architecture/msa/API%20Gateway/img.png)

## 1. 정의
API 게이트웨이는 모든 클라이언트 요청을 받아 적절한 마이크로서비스로 라우팅하거나,
여러 서비스에 병렬 요청을 수행해 응답을 조합해 반환하는 프록시 계층이다.

기본적으로 단일 진입점을 가지고 있는데 이는
클라이언트는 개별 서비스 주소나 포트를 알 필요 없이 게이트웨이 하나만 호출하면 되게 해준다.
(각 프론트 타입에 따라서, 가령 Web과 Mobile의 경우에 별도의 API Gateway를 두기도 한다)

크게 두 가지 종류로 나눌 수 있다.

- 프록시/라우팅: 단순히 해당 서비스로 요청을 전달
- 파닝아웃(Fan‑out): 여러 서비스에 병렬 호출 후 응답을 조합해 단일 응답 생성

## 2. 주요 역할
### 1) 라우팅(Routing)   
Client에서 요청하는 URL 경로나 헤더, 파라미터에 따라서 해당 하는 Micro service에 요청을 하는것이다.   
사실 이 부분은 기존에 모놀리식 방식에서도 사용하는 방식인데, 리버스 프록시로 웹을 구성할때 많이 보이는 방식이며
이렇게 구성된 경우 MSA로 수정할 때 클라이언트측에서는 변화를 느끼지 못하기 때문에 변경하기에 용이하다.

### 2) API 조합(API Composition)
한 번의 클라이언트 호출로 여러 서비스의 데이터를 합쳐 응답한다.   
이는 클라이언트와 각각의 마이크로 서비스간의 통신량을 줄이는데 용이하다.
왜 이런 통신량을 줄여야할까? 해당 내용은 [출처](https://learn.microsoft.com/ko-kr/dotnet/architecture/microservices/architect-microservice-container-applications/direct-client-to-microservice-communication-versus-the-api-gateway-pattern) 에서는 애매하게 
표기되어있는데, 내 생각을 덧붙여보자면 backend는 VPC(Virtual Private Cloud)로 구성되어있을 가능성이 있고(사실상 100%다)
이 경우 같은 리전에서 게이트웨이와 마이크로 서비스가 통신할 가능성이 높다. AWS의 경우 다른 가용영역의 경우 과금이 되지만 같은 가용영역일
경우 과금이 안되며, VPC에서 Outbound traffic에 대해서만 과금이 되기 때문에 사실상 traffic을 줄여서 비용을 줄이는 역할을 할 수 있는 것이다.

### 3) 보안(Security)
인증(Authentication)·인가(Authorization)를 중앙화하여 개별 서비스의 부담을 경감한다.
서비스를 분리할 경우 각각에 대한 인증을 따로 구현해야하는 골치아픈 상황이 있지만 아예 중앙화해버리면 한곳에서 관리하면 되므로
부담이 덜하다.

### 4) 부하 분산(Load Balancing) & 장애 격리(Fault Isolation)
서비스 인스턴스 풀에 트래픽 분산, 서킷 브레이커(Circuit Breaker)를 적용 가능하다.   
API Gateway 패턴을 취하면 클라이언트는 backend가 어떻게 구성되어있는지 모른다. 따라서 서비스 요청을 좀 더 유연하게
라우팅하고 로드를 밸런스있게 분산할 수 있기 때문에 가용성에 도움이 된다.   
또한 MSA의 경우 장애가 퍼질수있는데, 가령 특정 마이크로 서비스 A에서 다른 마이크로 서비스 B로 요청해서 처리해야하는 의존성있는
작업이 있다고 해보자, B에서 장애가 난다면 A에서 계속 요청을 보낼 것이고, B가 답신하지 않기 때문에 A의 경우 계속해서 요청 후 응답을
기다리는 스레드가 쌓일 것이다. 이 경우 A 역시 먹통이 될 것이므로 문제가 생기는데 이때 차단해버리는거이 서킷 브레이커이다.   
모놀리식의 경우 얄짤 없이 셧다운이지만 MSA에서는 최소의 장애만으로 서비스를 지킬 수 있는 것이다.

### 5) 로깅·모니터링(Observability)
요청·응답 메트릭, 에러 로깅을 집중해서 수집할 수 있다.    

오고 가는 요청의 창구가 하나이니, 해당 Gateway에서 모두 수집하면 분류 또한 편해진다.

## 3. 장점
### 1) 클라이언트 단순화
각 서비스 주소를 몰라도 되고, 호출 로직이 경량화된다. 이는 보안적으로 좀 좋기도하다.

### 2) API 최적화
모바일·웹·서드파티 등 클라이언트별로 최적화된 API 인터페이스 제공(Backends for Frontends)할 수 있다.   

사실 이건 좀 구현의 차이라고 볼 수 있는데, Gateway를 분리하면 구현하기 편해지는건 사실이긴하다.

### 3) 교차 관심사(Cross‑cutting Concern) 처리
보안, 로깅, 버전 관리, 요청 제한(Rate Limiting) 등을 중앙에서 관리할 수 있다.

### 4) 버전 관리(Versioning) 지원
API 버전 변경 시 게이트웨이만 수정해 레거시·신규 버전을 동시에 운영 가능하다.

이건 모놀리식에서 MSA로 변환할때도 유용하다.

## 4. 단점
### 1) 추가적 네트워크 홉
요청이 게이트웨이를 거치며 지연(latency)이 소폭 증가, 이는 트래픽 발생으로 추가적인 비용 발생으로 이어지기도 한다.

앞서 설명했던 같은 리전내의 다른 가용영역간 통신이 추가 비용 발생으로 이어진다.   
사실 이는 가용성 유지로 인해 어쩔수없는 부분이긴하다. 모놀리식 방식에서도 일반적으로 가용성을 위해서 서비스 운용 로직을
복제하여 각각 다른 가용영역에 두는게 일반적이기 때문이다.   

### 2) 복잡도 증가
게이트웨이 자체가 또 하나의 마이크로서비스로 개발·배포·운영·모니터링해야 한다.

### 3) 단일 장애점(SPOF)
게이트웨이 장애 시 전체 시스템 접근이 불가능해지므로, 고가용성(HA) 구성 필요하다.


> ※ 추가 업데이트 및 검증 예정이다.
{: .prompt-tip }

# 참고문헌
- [Microservices.io - api gateway pattern](https://microservices.io/patterns/apigateway.html)
- [AWS - API 게이트웨이 패턴](https://docs.aws.amazon.com/ko_kr/prescriptive-guidance/latest/modernization-integrating-microservices/api-gateway-pattern.html)
- [API 게이트웨이 패턴과 직접 클라이언트-마이크로 서비스 통신 비교](https://learn.microsoft.com/ko-kr/dotnet/architecture/microservices/architect-microservice-container-applications/direct-client-to-microservice-communication-versus-the-api-gateway-pattern)
- [Nginx store - 마이크로서비스 구축을 위한 API Gateway 패턴 사용하기](https://nginxstore.com/blog/api-gateway/%EB%A7%88%EC%9D%B4%ED%81%AC%EB%A1%9C%EC%84%9C%EB%B9%84%EC%8A%A4-%EA%B5%AC%EC%B6%95%EC%9D%84-%EC%9C%84%ED%95%9C-api-gateway-%ED%8C%A8%ED%84%B4-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0/)
- [Hudi blog - MSA 환경에서 장애 전파를 막기 위한 서킷 브레이커 패턴](https://hudi.blog/circuit-breaker-pattern/)
