---
title: OPENSTACK - Designate
author: blakewoo
date: 2025-1-23 16:50:00 +0900
categories: [Openstack]
tags: [Openstack, Designate] 
render_with_liquid: false
---

# Designate

## 1. 개요

> Designate에 대해 말하기에 앞서 DNS에 대해서 알아야한다.   
혹시 DNS에 대해서 모른다면 [OSI 7 Layer - 네트워크](https://blakewoo.github.io/posts/OSI-7-Layers-%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC%EA%B3%84%EC%B8%B5/)
의 DNS 설명을 읽어보고 오길 추천한다.
{: .prompt-tip }

Designate는 사용자와 운영자가 REST API를 통해 DNS 레코드, 이름 및 영역을 관리하고 기존 DNS 이름 서버를 구성하여
해당 레코드를 포함할 수 있는 OpenStack 서비스이다. Designate는 운영자가 OpenStack Network Service(Neutron)와 Compute
Service(Nova)와 통합하도록 구성할 수도 있으므로 플로팅 IP와 컴퓨트 인스턴스가 각각 생성될 때 레코드가 자동으로 생성되고 사용자 관리를 위해
OpenStack Identity Service(Keystone)를 사용한다. DNS 이름 서버의 소프트웨어 구현이 다양하기 때문에 Designate에는 플러그형 백엔드가
있어 이를 구성하여 대부분을 관리할 수 있으며, 특히 BIND9와 PowerDNS가 있다.

## 2. 구조
Designate 공식 문서에 나와있는 구조는 아래와 같다.

![img.png](/assets/blog/openstack/designate/img.png)    
출처 : https://docs.openstack.org/designate/latest/_images/Designate-Arch.png

### 1) Designate API
designate-api는 표준 OpenStack 스타일 REST API 서비스를 제공하여 HTTP 요청을 수락하고 Keystone으로 인증 토큰을 검증하고
AMQP를 통해 Designate Central 서비스로 전달한다. API의 여러 버전을 호스팅할 수 있으며 API 확장도 가능하여 핵심 API에
플러그형 확장을 허용한다.

designate-api는 HTTPS 트래픽을 처리할 수 있지만 일반적으로는 앞에 nginx와 같은 별도의 프록시 서버를 통해
HTTPS 통신을 처리하게 하는게 일반적이다.

### 2) Designate Central
designate-central는 메세지 큐를 통해 RPC 요청으로 제어되는 서비스이다. 데이터의 영구 저장소를 조정하ㄴ고 API의 데이터에 비즈니스 로직을 적용
한다. 저장소는 플러그인을 통해 제공되며, 기본적으로는 SQLAlchemy이지만 MongoDB 또는 다른 저장소 드라이버도 가능하다.

### 3) Designate MiniDNS
designate-miniDNS는 DNS NOTIFY를 보내고 영역 전송(AXFR) 요청에 응답하는 서비스이다.
이를 통해 Designate는 이러한 표준적인 통신 방법을 지원하는 모든 DNS 서버와 통합할 수 있다.
또한 Designate가 수행하는 모든 다른 형태의 DNS 프로토콜을 캡슐화한다.
예를 들어, 변경 사항이 라이브인지 확인하기 위해 SOA(Start of Authority, 인증정보) 쿼리를 보내는 것이 있다.

### 4) Designate Worker
designate-worker는 Designate가 관리하는 DNS 서버 상태와 기타 시간이 걸리는 작업 또는 복잡한 작업을 관리하는 서비스이다.
designate-worker는 pools.yaml 파일을 통해 Designate 데이터베이스에서 DNS 서버 구성을 읽는다.
이러한 DNS 서버 백엔드는 designate-worker에 로드되어 각 DNS 서버에서 영역과 레코드 세트를 생성, 업데이트 및 삭제할 수 있다.
작업자는 DNS 서버 Pool을 완벽하게 알고 있으므로 단일 designate-worker 프로세스가 여러 DNS 서버 Pool을 관리할 수 있다.

### 5) Designate Producer
designate-producer는 장기 실행 및 잠재적으로 큰 작업의 호출을 처리하는 서비스이다.
designate-producer 프로세스는 Designate가 관리하는 영역의 자동 할당된 샤드에 대한 작업을 시작하는데, 이러한
샤드는 영역 ID(UUID 필드)의 처음 세 문자를 기준으로 할당된다.
단일 designate-producer 프로세스가 관리하는 샤드 수는 총 샤드 수를 designate-producer 프로세스 수로 나눈 값과 같기 때문에
designate-producer 프로세스가 많이 시작될수록 한 번에 생성되는 작업이 줄어든다.

designate-producer에 현재 구현된 작업으로는 Ceilometer에 대한 dns.zone.exists 이벤트 배포, 데이터베이스에서 삭제된 영역 제거,
새로 고침 간격에 따라 보조 영역 폴링, 지연된 NOTIFY 트랜잭션 생성, 오류 상태인 영역의 주기적 복구 호출 등이 있다.

### 6) Designate Sink
designate-sink는 이벤트 알림을 수신하는 선택적 서비스이다. ```ex) Compute.instance.create.end```
Nova 및 Neutron에 대한 핸들러가 제공되며 알림 이벤트를 사용하여 레코드 생성 및 삭제를 트리거할 수 있다.

현재 designate-sink는 handler-nova 구성에 지정된 형식을 사용하여 간단한 A 레코드를 생성하며,
이벤트 알림의 모든 필드는 레코드를 생성하는 데 사용할 수 있다.

### 7) DNS Backend
백엔드는 특정 DNS 서버의 드라이버이다. Designate는 PowerDNS, BIND, NSD, DynECT 등 여러 백엔드 구현을 지원하며,
필요에 맞게 자체 백엔드를 구현할 수도 있고, 기존 백엔드를 보완하는 추가 기능을 제공하는 확장 기능도 있다.

### 8) Message Queue
Designate는 구성 요소 간 메시징에 oslo.rpc를 사용하므로 지원되는 메시징 버스(예: RabbitMQ, Qpid 또는 ZeroMQ)에
대한 요구 사항을 상속한다. 일반적으로 이는 RabbitMQ 설정이 Designate에 맞춰져 있음을 의미하지만 일반 설치에는 단일 가상 호스트만
필요하므로 적합하다고 생각되는 다른 RabbitMQ 인스턴스를 자유롭게 사용할 수 있다.

### 9) Database/Storage
스토리지 드라이버는 특정 SQL/NoSQL 서버를 위한 드라이버이다. Designate는 데이터의 영구 저장을 위해 SQLAlchemy
지원 스토리지 엔진이 필요하다. 권장 드라이버는 MySQL이다.


> ※ 추가 업데이트 예정
{: .prompt-tip }


# 참고문헌
- [오픈스택 - Designate 소개](https://docs.openstack.org/designate/latest/intro/index.html)
- [오픈스택 - Designate 구조](https://docs.openstack.org/designate/latest/contributor/architecture.html)
