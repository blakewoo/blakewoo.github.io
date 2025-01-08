---
title: Devstack 설치 및 간단한 운용
author: blakewoo
date: 2025-1-8 17:40:00 +0900
categories: [Trial and error]
tags: [Openstack, Devstack] 
render_with_liquid: false
---

# Devstack

> ※ 본 포스팅은 2025년 1월 8일 기준으로 작성되었음
{: .prompt-tip }

## 1. Devstack이란?
```
DevStack은 git master의 모든 최신 버전을 기반으로 완전한 OpenStack 환경을 빠르게 구축하는 데 사용되는 일련의 확장 가능한 스크립트입니다.
개발 환경으로 대화형으로 사용되며 OpenStack 프로젝트의 많은 기능 테스트의 기초로 사용됩니다.
```

오픈 스택 공식 웹페이지에서 설명하는 Devstack이란 무엇인가에 대한 설명이다. 기본적으로 아래의 서비스를 포함하고 있다.
- Keystone : 인증
- Swift : 객체 스토리지
- Glance : 이미지 서비스
- Cinder : 블록 스토리지
- Nova : 컴퓨팅 가상화 관리 서비스
- Placement : 컴퓨팅 지표
- Neutron : 네트워킹
- Horizon : 대시 보드

그 외의 패키지는 아래와 같은 서비스를 이용한다
- MySQL : DB 
- Rabbit : Messaging queue
- Apache : Webserver

이 정도의 서비스를 설치해두면 당장 OPENSTACK을 작게 운용하는데 문제가 없다는 뜻이다.
DEVSTACK에 대해서 대략 알아보았으니 어떻게 설치했는지 알아보겠다.

## 2. 설치 절차

### 1) 기본 환경
먼저 설치를 진행한 환경은 아래와 같다.
```
OS: UBUNTU 24.04.1 LTS
CPU: INTEL(R) CORE(TM) i7-8700K CPU @ 3.7GHZ 
RAM: 16GB (2113MHz)
STORAGE TYPE: SATA SSD 256GB
```

### 2) 절차

#### a. Devstack을 위한 환경 준비
필요한 몇가지 패키지를 미리 설치해주면 좋다.
미리 루트 계정으로 전환해두는게 작업시 편하다.
```shell
apt-get update
# apt 패키지 업데이트
apt-get install -y git
# git clone해야하므로 git 설치
apt-get install -y vim
# vim 에디터 설치
apt-get install -y net-tools
# ifconfig 쓰기 위함
```

#### b. 설치
나머지는 devstack quick start에 나와있는대로 진행하면 된다.
사용자를 손수 만들 수 있지마 devstack에서 제공하는대로 사용할 수도 있다.
```shell
cd /home
# home 경로로 이동
git clone https://opendev.org/openstack/devstack
cd devstack
# devstack git을 clone해옴
./tools/create-stack-user.sh
# 유저 추가
cd ..
chown -R stack devstack
# 생성한 유저에게 git으로 받아온 폴더내 전체의 소유권 이전
sudo -u stack -i
cd /home/devstack
# 유저 변경 및 폴더로 이동
```

이후 ifconfig 명령어로 현재 ip를 체크해둔다
이후 vim 편집기로 local.conf 파일을 만들어서 아래와 같이 입력한다.
```
[[local|localrc]]
ADMIN_PASSWORD=secret
HOST_IP={아까 확인한 IP}
DATABASE_PASSWORD=$ADMIN_PASSWORD
RABBIT_PASSWORD=$ADMIN_PASSWORD
SERVICE_PASSWORD=$ADMIN_PASSWORD
```

입력 후 저장한다음에 아래의 쉘을 시작한다. 

```
./stack.sh
```

#### d. 구동 확인

웹 브라우저를 하나 켜서 localhost로 접속하거나 ifconfig로 접속한 ip로 접속하면 Dashboard가 뜬다.

![img.png](/assets/blog/trial_error/devstack/install/img.png)

#### ※ 트러블 슈팅

Q. localhost나 ip로 접속했는데 Dashboard 화면이 뜨지 않는다.   

A. 그냥 다시 까는게 속시원하다.   
devstack 프로젝트 안에는 stack.sh 말고도 devstack을 중단시키는 unstack.sh과
설치된 devstack을 지우는 clean.sh이 포함되어있다.
만약 devstack이 구동 중이라면 아래와 같은 절차로 삭제하고 다시 설치하면 된다.
```shell
./unstack.sh
./clean.sh
./stack.sh
```

Q. 뜨긴 뜨는데 font가 깨진다.

A. apache2에서 라우팅하는 부분을 수정해주면 된다.
horizon에서 해당 부분의 경로처리가 잘못되어서 생기는 문제다.
기본적으로 devstack은 apache2를 이용해서 각 요청들을 라우팅한다.
따라서 devstack의 routing 설정을 임시로 잡아준다면 font를 제대로 불러올 수 있다.

"/etc/apache2/sites-available/horizon.conf" 경로의 파일을 보면 다음과 같은 부분이 있다.

```
... ...
Alias /dashboard/static /opt/stack/horizon/static
... ...
```

여기에 다음 문구를 추가해준다.

```
Alias /static /opt/stack/horizon/static
```

font 데이터를 "/dashboard/static"이 아닌 "/static"로 요청해서 생기는 문제이므로
위와 같이 바꿔준다면 당장은 사용할 수 있다.

> ※ 본 해결법은 임시 조치에 가깝다. 문제 파악 후 좀 더 확실한 해결법을 업데이트 할 예정이다.
{: .prompt-tip }


# 참고문헌
- [OPENSTACK - DEVSTACK](https://docs.openstack.org/devstack/latest/index.html)
