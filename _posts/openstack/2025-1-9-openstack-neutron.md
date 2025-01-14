---
title: OPENSTACK - Neutron
author: blakewoo
date: 2025-1-10 15:00:00 +0900
categories: [Openstack]
tags: [Openstack, neutron] 
render_with_liquid: false
---

# Neutron

> ※ 아직 해당 포스팅 작성이 완료되지 않았으므로 참고만 하기 바람
{: .prompt-tip }

## 1. 개요
이전에 포스팅했던 nova가 가상화머신을 관리하는 서비스라면 이번에 포스팅할 neutron의 경우에는
가상 네트워크 인프라의 구성과 관리를 담당하는 서비스이다.
API를 통해 가상 네트워크, 서브넷, 라우터, 포트 등의 네트워킹 리소스를 정의하고 관리할 수 있다.

## 2. 구성요소

### 1) Neutron-server
API 요청을 수락하고 해당 작업을 위해 적절한 OpenStack Networking 플러그인으로 라우팅한다.

### 2) Openstack Networking plug-ins and agents
포트를 연결 및 연결 해제하고, 네트워크 또는 서브넷을 생성하고, IP 주소 지정을 제공한다.
이러한 플러그인과 에이전트는 특정 클라우드에서 사용되는 공급업체와 기술에 따라 다르다.
OpenStack Networking은 Open vSwitch, Linux 브리징, Open Virtual Network(OVN),
SR-IOV 및 Macvtap용 플러그인과 에이전트와 함께 제공된다.

일반적인 에이전트로는 L2(2계층), L3(3계층), DHCP 등이 있다.

- L2 agent : 일반적으로 네트워크와 컴퓨터 노드에 설치되며 RPC를 사용해 neutron-server 와 통신한다.
  L2 에이전트는 디바이스 추가 또는 삭제되는 상황을 모니터링 하며 도중에 문제가 생길 경우 이를 전달하고 호스트상의
  네트워크를 설정하는 역할도 한다. 또한 linux bridge, ovs(open vswitch), 보안그룹 및 vlan 태깅도 처리할 수 있다.


- L3 agent : 네트워크 노드에 위치하며 neutron-server로부터 라우터 관리, 라우팅, 플로팅 IP에 대한 메세지를 받아서   
  관리한다. 각 내부 네트워크 간에 데이터를 전달하고 내부 네트워크 정보를 받아 외부 네트워크로 전달하는 역할도 수행한다.


- DHCP agent : IP 주소 할당에 사용되며, neutron-server로 부터 네트워크 생성 및 삭제에 대한 메세지를 받으면
  dnsmasq 기능을 사용해 DHCP 서버로 사용된다.


- metadata agent : 인스턴스 내부 클라이언트 metadata 요청을 nova metadata 서비스에 전달하며 일반적으로
  네트워크 노드에 설치된다. RPC를 통해 neutron-server와 통신하며 IP주소, 호스트 이름, 프로젝트와 같이 인스턴스가
  요청한 정보를 제공하는 역할도 한다.


### 3) Messaging queue
대부분의 OpenStack Networking 설치에서 Neutron-Server와 다양한 에이전트 간의 정보를 라우팅하는 데 사용된다.
또한 특정 플러그인의 네트워킹 상태를 저장하는 데이터베이스 역할도 한다.

## 3. 소스코드 구조
기본적으로 Openstack neutron의 소스코드는 [이곳](https://releases.openstack.org/teams/neutron.html) 에서
받을 수 있다. 릴리즈 버전들을 올려둔 공식 사이트이다.
소스코드를 받아와보면 여러 폴더들이 있다.

실질적인 소스는 neutron 폴더에 포함되어있으며 내용은 아래와 같다. 

### agent
- **역할** : 네트워크 에이전트와 관련된 코드.
- **내용** : DHCP 에이전트, L3 에이전트, Open vSwitch 에이전트와 같은 네트워크 에이전트의 구현. 각 에이전트의 동작 로직과 관리 코드 포함.

### api
- **역할** : Neutron의 API와 관련된 코드.
- **내용** : RESTful API 엔드포인트 정의, 요청 처리 로직, JSON 직렬화/역직렬화 포함.

### cmd
- **역할** : Neutron 실행 명령어와 관련된 코드.
- **내용** : CLI 도구 및 서비스 시작 스크립트 (`neutron-server`, `neutron-ovn-db-sync` 등).

### common
- **역할** : 공통적으로 사용되는 유틸리티 함수와 클래스.
- **내용** : 로깅, 설정, 데이터 구조 관련 코드.

### conf
- **역할** : 설정과 관련된 코드.
- **내용** : Neutron 설정 옵션 정의 및 기본값 지정.

### core_extensions
- **역할** : Neutron의 코어 확장 기능.
- **내용** : 플러그인 또는 기본 네트워크 기능을 확장하는 모듈.

### db
- **역할** : 데이터베이스 관련 코드.
- **내용** : 데이터 모델 정의, CRUD 연산, SQLAlchemy ORM 매핑.

### exceptions
- **역할** : Neutron에서 사용하는 예외 클래스 정의.
- **내용** : 에러 핸들링 및 사용자 정의 예외 정의.

### extensions
- **역할** : Neutron 확장 기능 정의.
- **내용** : API 확장 및 기능 플러그인 인터페이스 정의.

### hacking
- **역할** : 코드 스타일 검사 및 가이드라인.
- **내용** : PEP8 및 OpenStack 프로젝트의 코드 스타일 규칙 정의.

### ipam
- **역할** : IP 주소 관리와 관련된 코드.
- **내용** : 서브넷 및 IP 주소 할당 로직.

### locale
- **역할** : 다국어 지원과 관련된 번역 파일.
- **내용** : `.po`, `.mo` 파일 및 지역화된 문자열 포함.

### notifiers
- **역할** : 알림 및 메시징과 관련된 코드.
- **내용** : 이벤트 기반 알림 시스템 구현.

### objects
- **역할** : Neutron의 객체 모델.
- **내용** : RPC와 데이터베이스 간의 데이터 전송에 사용되는 객체 정의.

### pecan_wsgi
- **역할** : Pecan 기반 WSGI 애플리케이션 코드.
- **내용** : Pecan 프레임워크를 사용한 API 구현.

### plugins
- **역할** : Neutron 플러그인 관련 코드.
- **내용** : 네트워크 백엔드 구현(예: Open vSwitch, OVN, ML2 등).

### privileged
- **역할** : 권한 상승된 작업과 관련된 코드.
- **내용** : 루트 권한이 필요한 네트워크 작업 처리.

### profiling
- **역할** : 성능 프로파일링 및 디버깅 도구.
- **내용** : 요청 추적 및 성능 분석 코드.

### quota
- **역할** : 할당량(Quota) 관리와 관련된 코드.
- **내용** : 네트워크, 서브넷, 포트 등의 자원 제한 로직.

### scheduler
- **역할** : 스케줄러와 관련된 코드.
- **내용** : L3 에이전트 스케줄링 로직 포함.

### server
- **역할** : Neutron 서버와 관련된 코드.
- **내용** : `neutron-server` 실행 로직 및 서비스 등록 코드.

### services
- **역할** : 서비스 플러그인과 관련된 코드.
- **내용** : LBaaS, FWaaS, VPNaaS 같은 네트워크 서비스 구현.

### tests
- **역할** : 테스트 코드.
- **내용** : 단위 테스트, 기능 테스트, 통합 테스트 포함.

### wsgi
- **역할** : WSGI 서버와 관련된 코드.
- **내용** : HTTP 요청 처리, 라우팅 및 미들웨어 정의.




# 참고문헌
- [오픈스택 - neutron](https://docs.openstack.org/neutron/latest/)
- [Somaz의 IT 공부 일지 - Openstack Neutron이란?](https://somaz.tistory.com/123)
