---
title: OPENSTACK - Glance
author: blakewoo
date: 2025-1-15 15:30:00 +0900
categories: [Openstack]
tags: [Openstack, cinder] 
render_with_liquid: false
---

# Glance

> ※ 아직 해당 포스팅 작성이 완료되지 않았으므로 참고만 하기 바람
{: .prompt-tip }

## 1. 개요
OS 이미지를 이용해서 새로운 볼륨을 만들 수 도 있고 현재 운용중인 볼륨을 이미지로 만들 수도 있는 서비스이다.
AWS에서의 AMI를 생각하면 편하다.

## 2. 구성요소
### 1) glance-api
Glance API를 제공하는 서버 데몬이다.

### 2) glance-cache-cleaner
이미지를 캐싱하는 동안 시간 초과 및 예외 발생 등으로 문제가 발생하게 되면 완전하지 않은 잘못된 캐싱이 만들어지는데
이런 항목을 디버깅 목적으로 남겨두긴 하지만 life time을 두고 일정시간 이후에 지워야한다.
이를 위해 cron에서 주기적인 작업을 실행되게 설계된 것이다.

### 3) glance-cache-manage
캐싱을 관리할 수 있는 유틸리티이다.

### 4) glance-cache-prefetcher
이는 미리 준비할 이미지를 대기열에 넣은 후 명령줄에서 실행하는 유틸리티이다.

### 5) glance-cache-pruner
공간이 image_cache_max_size 구성 옵션에 설정된 값을 초과하면 Glance 캐시에서 이미지를 정리하게되는데
이러한 작업을 하는 유틸리티이며 30분마다 주기적으로 처리하게 되어있다.

### 6) glance-control
Glance 데몬을 시작, 중지, 재시작할 수 있게 해주는 유틸리티이다.

### 7) glance-manage
glance-manage는 Glance 설치를 관리하고 구성하는 유틸리티이다.
glance-manage의 중요한 용도 중 하나는 데이터베이스를 설정하는 것으로 데이터베이스와
현재 Glance 서비스 상태를 싱크하는 역할을 한다.

### 8) glance-replicator
glance-replicator는 기존 glance 서버에 저장된 이미지를 사용하여 새로운 glance 서버를 만드는 데 사용할 수 있는 유틸리티이다.
복제된 glance 서버의 이미지는 원본의 uuid, 메타데이터 및 이미지 데이터를 보존한다.

### 9) glance-scrubber
Glance 이미지의 경우 실질 이미지는 백엔드 서버에, 메타 데이터는 db에 저장되는데 이미지 삭제 요청을 내릴 경우
서버에서 이미지를 삭제 후 db에 반영되는 식이다. 이 경우 이미지 크기가 너무 크다면 이미지 삭제 시간이 너무 오래 걸려
timeout 등으로 인해 서비스 품질에 문제가 있을 수 있다.
따라서 db에만 삭제 표기로 해두고 실제 이미지는 이후에 삭제하게끔 설정해둘 수 있다.
이럴 경우 glance-scrubber가 필요한 것이며 삭제 표기된 실제 이미지를 삭제하거나 혹은 삭제 표기는 되었지만 아직 삭제되지 않은
이미지를 복원하는데 사용할 수 있다.

### 10) glance-status
glance-status는 업그레이드를 방해할 수 있는 요소를 프로그래밍 방식으로 점검하여 운영자가
Glance를 업그레이드할 때 도움을 주는 명령줄 유틸리티이다.



# 참고문헌
- [오픈스택 - Glance](https://docs.openstack.org/glance/latest/)

