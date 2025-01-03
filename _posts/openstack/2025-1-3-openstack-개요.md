---
title: OPENSTACK - 개요
author: blakewoo
date: 2025-1-3 17:20:00 +0900
categories: [Openstack]
tags: [Openstack] 
render_with_liquid: false
---

# 오픈 스택(Open stack)

> ※ 관련 모듈의 추가적인 연구가 있다면 링크를 달 예정이기 때문에 잦은 업데이트가 있을 예정이다.
{: .prompt-tip }

## 1. 개요
오픈 스택이란 "데이터 센터 전체에서 대규모 컴퓨팅, 스토리지 및 네트워킹 리소스풀을 제어하는 클라우드 운영체제" <sup>1</sup> 이다.
다수의 모듈로 이루어져있으며 이 모듈들의 상호 동작 아래 리소스풀을 제어하게 된다.

## 2. 모듈에 대한 간략한 설명
### 1) Ironic (Bare Metal service)
베어메탈 서버를 위한 서비스이다. 이 베어메탈 서버라는 것은 가상화 서버에 반대되는 개념인데, 가상화되지 않고 사용하는 서버를 말한다.
가상화가 만들어지기 이전에 사용하던 방식을 말한다.
이는 오픈스택과 별도로 운용될 수도 있고, 오픈스택과 통합되어 운용될 수도 있다.

### 2) Cinder (Block Storage service)
Nova 가상 머신이나 Ironic 베어메탈 호스트나 컨테이너등 서비스를 위한 블록 스토리지 서비스이다.
AWS로 치면 EBS(Elastic Block Storage)에 해당하는 것으로 보인다.

### 3) Storlets (Compute inside Object Storage service)
Docker 컨테이너를 사용하여 안전하고 격리된 방식으로 객체 저장소 내에서 사용자 정의 코드를 실행할 수 있게 해주는
Openstack Swift의 확장 프로그램이다.

### 4) Nova (Compute service)
가상머신을 만들어주는 서비스이다. 컨테이너에 대한 지원은 제한되며 리눅스 서버에서 데몬으로 돌아간다.
AWS의 EC2와 같은 서비스로 이해하면 된다.

### 5) Zun (Containers service)
컨테이너를 위한 서비스이다. docker를 말하는 것이냐고 말할수 있는데, 실상 이 zun의 기본 컨테이너 엔진이
docker를 사용한다. 말하자면 docker 같은 컨테이너 엔진을 잘 사용할 수 있게 지원하는 서비스라고 할 수 있다.

### 6) Horizon (Dashboard)
대시보드를 위한 서비스이다. 웹 기반으로 제공되며 다른 서비스들이 잘 돌아가는지 일목요연하게 볼 수 있는 서비스라고 할 수 있다.

### 7) Trove (Database service)
데이터베이스를 손쉽게 사용할 수 있게하는 서비스이다. 관계형, 비관계형 상관없이 모두 서비스를 지원하며 image를 통해 instance를 생성하여
서비스를 제공한다. 

### 8) Designate (DNS service)
멀티 테넌트 DNSaaS 서비스로 keystone 인증이 있는 REST API를 제공한다. Nova나 Neutron 작업에 따라 레코드를 자동 생성하도록 구성할 수 있다.

### 9) Keystone (Identity service)
API 클라이언트를 인증하거나 서비스 검색이나 분산 다중 테넌트 권한 부여를 제공하는 서비스이다.

### 10) Glance (Image service)
이미지를 생성하고 검색하고 등록하는 기능을 제공하는 서비스이다.

### 11) Barbican (Key Manager service)
키 매니저 서비스로 비밀 데이터의 안전한 저장 및 관리를 제공한다. 대칭키, 비대칭 키, 인증서 및 원시 바이너리 데이터와 같은 자료가 포함되어있다.

### 12) Octavia (Load-balancer service)
로드밸런서 서비스이다. 수신한 요청들을 지정된 서버에 균등하거나 특정 조건에 따라 가중하여 전달해주는 서비스이다.

### 13) Zaqar (Messaging service)
멀티 테넌트를 위한 클라우드 메세지와 알림 서비스이다.

### 14) Neutron (Networking service)
네트워크 인터페이스간에 연결해주는 서비스이다. 가상네트워크나 서브넷, 라우터, 포트등 네트워킹 리소스를 정의하고 관리할 수 있다.
AWS에서 VPC나 SUBNET을 설정해줄 수 있는 서비스를 생각하면 편할 것 같다.

### 15) Tacker (NFV Orchestration service)
Tacker에 대해 말하기에 앞서 NFV에 대해서 알아야한다. NFV는 네트워크 기능을 가상화하는 기술로
새로운 장비를 설치하지 않아도 네트워크 기능을 구현할 수 있도록 하는 것이다. 이 네트워크 기능을 오케스트레이션 해주는 서비스이다.

### 16) Swift (Object Storage service)
오브젝트 스토리지 서비스이다. 고가용성이고 분산되어 운용될 수 있으며 객체/blob 들을 저장할 수 있다.
AWS의 S3를 생각하면 편할 것 같다.

### 17) Heat (Orchestration service)
오케스트레이션 서비스로, API 호출을 통해 템플릿을 통해 실행 중인 클라우드 애플리케이션을 생성할 수 있다.
쿠버네티스를 생각하면 편할 듯 하다.

### 18) Placement (Placement service)
리소스를 트래킹하는데 사용하는 서비스이다. 원래는 Nova 서비스에서 해당 인스턴스의 사용량을 트래킹하기 위해 만들어졌으나
다른 서비스에서 사용할 수 있도록 설계되었다.

### 19) Cloudkitty (Rating service)
Rating-as-a-Service로 정의되어있다. 지표 기반으로 Rating해주는데, 이를 이용해서 Pricing이나 Billing 같은 기능을 추가해서 사용할 수
있다.

### 20) Vitrage (RCA (Root Cause Analysis) service)
Openstack에서 발생하는 알람과 이벤트를 분석하여 문제의 근본 원인에 대한 정보를 제공하고
문제가 발생하기전에 알려줄 수 있는 근본 원인 분석 서비스이다.

### 21) Blazar (Resource reservation service)
리소스 예약 서비스이다. 가령 가상 머신을 에약하거나 호스트를 예약하거나 하는 식으로 미리 리소스를 예약해둘 수 있다.

### 22) Manila (Shared File Systems service)
공유 파일 시스템을 서비스로 제공하기 위한 공유 파일 시스템 서비스이다. 다른 서버간에 파일을 공유하기 위해 사용할 수 있다.
(정확한지는 잘 모르겠지만 NAS 정도로 생각하면 좋을 듯 하다)

### 23) Aodh (Telemetry Alarming services)
알람서비스로 Ceilometer 또는 Gnocchi가 수집한 지표나 이벤트 데이터에 대해 정의된 규칙에 따라 작업을 트리거하는 서비스를 제공한다.

### 24) Ceilometer (Telemetry Data Collection service)
Openstack의 핵심 구성요소에서 데이터를 정규화하고 변환하는 기능을 제공하는 데이터 수집 서비스이다.

### 25) Mistral (Workflow service)
워크플로 서비스로 특정 순서로 실행해야하는 여러 개의 상호 연결된 단계로 구성된 프로세스를 실행 할 수 있는 서비스이다.
다수의 서비스에서 실행해야하는 순차 서비스를 자동화 할 수 있는 서비스라고 생각하면 편할 듯 하다.

# 참고문헌
- 1. [오픈스택 공식 문서 서두](https://docs.openstack.org/2024.2/)
- 2. [오픈스택 공식 문서 - 각 프로젝트](https://docs.openstack.org/2024.2/projects.html)
