---
title: OPENSTACK - Cinder
author: blakewoo
date: 2025-1-14 16:50:00 +0900
categories: [Openstack]
tags: [Openstack, cinder] 
render_with_liquid: false
---

# Cinder

> ※ 아직 해당 포스팅 작성이 완료되지 않았으므로 참고만 하기 바람
{: .prompt-tip }

## 1. 개요
인스턴스가 사용할 수 있는 영구 블록 스토리지 서비스이다. AWS로 따지자면 EBS(Elastic Block Storage)로 생각하면 편하다.
(단, 공유 스토리지 서비스는 제공하지 않는다. 그건 다른 서비스를 사용해야한다)
Compute 인스턴스가 부팅 가능한 스토리지로 사용할 수도 있다.

제공하는 기능은 아래와 같다.

### 1) 백엔드 스토리지 디바이스
블록 스토리지 서비스는 서비스가 구축된 어떤 형태의 백엔드 스토리지를 필요로 한다. 기본 구현은 "cinder-volumes"라는 로컬 볼륨 그룹에서
LVM을 사용하는 것이다. 기본 드라이버 구현 외에도 블록 스토리지 서비스는 외부 RAID Array 또는 기타 Storage Appliance와 같이 활용할 다른
스토리지 장치에 대한 지원을 추가하는 수단도 제공한다. 이러한 백엔드 스토리지 장치는 하이퍼바이저로 KVM 또는 QEMU를 사용할 때 사용자 정의 블록
크기를 가질 수 있다.

### 2) 사용자와 테넌트
블록 스토리지 서비스는 역할 기반 액세스 할당을 사용하여 여러 클라우드 컴퓨팅 소비자 또는 고객(공유 시스템의 테넌트)이 사용할 수 있다.
역할은 사용자가 수행할 수 있는 작업을 제어한다. 기본 구성에서 대부분의 작업은 특정 역할이 필요하지 않지만 규칙을 유지 관리하는 cinder
정책 파일에서 시스템 관리자가 이를 구성할 수 있다.

### 3) 볼륨
컴퓨트 노드에 iSCSI를 통해 연결되는 영구적인 읽기/쓰기 블록 스토리지 장치로,
인스턴스의 보조 스토리지로 연결하거나 루트 스토어로 사용하여 인스턴스를 부팅할 수 있다.

### 4) 스냅샷
특정 시점에서의 볼륨의 읽기 전용 복사본이다. 현재 사용 중인 볼륨(옵션: --force True) 또는 사용 가능한 상태의 볼륨에서 생성할 수 있으며
스냅샷을 사용하여 새 볼륨을 생성할 수도 있다.

### 5) 백업
현재 Object Storage(swift)에 저장된 볼륨의 보관된 사본이다.


## 2. 구성요소

### 1) cinder-api
Block Storage 서비스 전체에서 요청을 인증하고 라우팅하는 WSGI 앱이다. OpenStack API만 지원하지만,
Block Storage 클라이언트를 호출하는 Compute의 EC2 인터페이스를 통해 수행할 수 있는 변환이 있다.

### 2) cinder-scheduler
적절한 볼륨 서비스에 대한 요청을 예약하고 라우팅한다. 구성에 따라 실행 중인 볼륨 서비스에 대한 간단한 라운드 로빈 스케줄링이 될 수도 있고,
필터 스케줄러를 사용하여 더 정교해질 수도 있다. 필터 스케줄러는 기본값이며 용량, 가용성 영역, 볼륨 유형 및 기능과 같은 항목에 대한 필터와
사용자 지정 필터를 활성화한다.

### 3) cinder-volume
블록 스토리지 장치, 특히 백엔드 장치 자체를 관리한다.

### 4) cinder-backup
블록 스토리지 볼륨을 OpenStack Object Storage(swift)에 백업하는 수단을 제공한다.

## 3. 소스코드 구조
기본적으로 Openstack cinder의 소스코드는 [이곳](https://releases.openstack.org/teams/cinder.html) 에서
받을 수 있다. 릴리즈 버전들을 올려둔 공식 사이트이다.
소스코드를 받아와보면 여러 폴더들이 있다.

실질적인 소스는 cinder 폴더에 포함되어있으며 내용은 아래와 같다. 

### api
- **역할** : API 처리
- **내용** : Cinder의 API 요청을 처리하는 모듈로 외부 요청을 수신하고 처리한다.

### backup
- **역할** : 블록 스토리지 백업
- **내용** : 블록 스토리지 볼륨 백업에 관련된 기능을 처리하며 Object Storage(Swift)를 활용하여 백업을 관리한다.

### brick
- **역할** : 블록 스토리지 장치 연결
- **내용** : iSCSI와 같은 블록 스토리지 장치 연결 및 데이터 전송을 관리했었는데, os-brick이라는 pypi library로 이관되었다.

### cmd
- **역할** : 명령줄 유틸리티
- **내용** : Cinder 서비스 실행에 사용되는 명령줄 유틸리티를 포함한다. 예: `cinder-volume`, `cinder-api` 실행 명령

### common
- **역할** : 공통 유틸리티와 그외 잡다한 라이브러리
- **내용** : 통신을 위한 패키지나 cors를 방지하는 미들웨어 같은 공용 유틸리티가 포함되어있다.

### compute
- **역할** : Nova 서비스와 통합
- **내용** : Nova 서비스와 같이 사용할 수 있게 연결해주는 코드가 포함되어있다.

### db
- **역할** : DB Access
- **내용** : 스냅샷, 볼륨 생성 및 삭제 등에 대해 DB에 기재 및 읽어들이기 위한 코드

### group
- **역할** : 볼륨 그룹
- **내용** : 그룹 볼륨 매니저와 상호 작용하기 위한 코드

### image
- **역할** : Glance 서비스와 통합
- **내용** : Glance의 image와 volume 서비스를 연결하기 위한 코드

### interface
- **역할** : 인터페이스 정의
- **내용** : 볼륨 드라이버와 백업 드라이버, fibre channel zone manager 드라이버의 interface 정의

### keymgr
- **역할** : 키 매니저와의 연동
- **내용** : Cinder 서비스와 키 매니저의 연동을 위한 코드

### locale
- **역할** : 다국어 지원
- **내용** : 다국어 지원을 위한 번역 파일이 포함되어있다.

### message
- **역할** : 사용자 메세지 관리 기능
- **내용** : 사용자의 작업에 따라 미리 정의된 메세지를 전달한다. 에러가 났을 때 어디서 에러가 났는지 알 수 있다.

### objects
- **역할** : 각 DB 데이터 구조 정의
- **내용** : 데이터베이스 모델과 연결된 ORM(Object-Relational Mapping) 클래스 및 Cinder 데이터 구조를 정의한다.

### policies
- **역할** : 정책 파일과 권한 제어를 담당
- **내용** : 정책 파일과 권한 제어를 담당한다. 예를 들자면 액세스 권한 및 작업 제한 정의 같은 것들이 있다.

### privsep
- **역할** : 시스템에 관련된 부분을 처리
- **내용** : nvmet, scst, tgt, cgroup, format_inspector, fs, lvm, path에 관련된 부분을 처리한다.

### scheduler
- **역할** : 볼륨 작업 스케줄러
- **내용** : 요청된 볼륨 생성 및 작업을 적절한 스토리지 노드로 라우팅하는 스케줄링 작업을 수행한다.

### tests
- **역할** : 테스트 코드
- **내용** : 테스트 하기 위한 코드들

### transfer
- **역할** : 볼륨 전송
- **내용** : 볼륨 전송(소유권 이전)과 관련된 작업을 처리한다.

### volume
- **역할** : 블록 스토리지 볼륨 관리
- **내용** : 블록 스토리지 볼륨의 생성, 삭제, 연결 및 복제 작업을 관리한다.

### wsgi
- **역할** : WSGI 서버와 관련된 코드
- **내용** : Cinder의 WSGI 애플리케이션을 관리하며 API 요청의 진입점을 제공한다.

### zonemanager
- **역할** : Fibre Channel zone manage
- **내용** : Fibre Channel zone 관리 작업을 처리한다.


# 참고문헌
- [오픈스택 - Introduction to the Block Storage service](https://docs.openstack.org/cinder/latest/configuration/block-storage/block-storage-overview.html)
- [오픈스택 - cinder 소스코드](https://tarballs.openstack.org/cinder/cinder-25.0.0.tar.gz)
