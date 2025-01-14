---
title: OPENSTACK - Cinder
author: blakewoo
date: 2025-1-14 15:00:00 +0900
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




# 참고문헌
- [오픈스택 - Introduction to the Block Storage service](https://docs.openstack.org/cinder/latest/configuration/block-storage/block-storage-overview.html)

