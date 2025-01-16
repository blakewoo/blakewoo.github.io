---
title: OPENSTACK - Keystone
author: blakewoo
date: 2025-1-16 16:00:00 +0900
categories: [Openstack]
tags: [Openstack, keystone] 
render_with_liquid: false
---

# Keystone

## 1. 개요
Keystone이란 Openstack에서 Identity API를 구현하여 API 클라이언트 인증, 서비스 검색, 분산 다중 테넌트 권한 부여를
제공하는 서비스이다.

Keystone은 하나 이상의 엔드포인트에서 노출되는 내부 서비스 그룹으로 구성되는데,
이러한 서비스 중 다수는 프론트엔드에서 조합하여 사용됩니다. 예를 들어, 인증 요청(authenticate call)은
Identity 서비스로 사용자/프로젝트 자격 증명을 확인하고, 성공 시 Token 서비스를 통해 토큰을 생성 및 반환한다.

## 2. 구성요소

### 1) Identity
Identity 서비스는 인증 자격 증명을 검증하고 사용자와 그룹에 대한 데이터를 제공한다.
기본적으로 이 데이터는 Identity 서비스에 의해 관리되며 관련 CRUD 작업을 처리할 수 있다.
그러나 더 복잡한 경우, 데이터는 권위 있는 백엔드 서비스에 의해 관리됩니다.
예를 들어, Identity 서비스가 LDAP(Lightweight Directory Access Protocol)의 프론트엔드 역할을 하는 경우,
LDAP 서버가 데이터의 신뢰 가능한 소스가 되며 Identity 서비스는 해당 정보를 중계하는 역할을 한다.

#### a. 사용자(Users)
사용자는 개별 API 소비자를 나타낸다. 사용자는 특정 도메인에 소속되어야 하며
도메인 내에서만 고유한 이름을 갖는다.

#### b. 그룹(Groups)
그룹은 사용자 모음을 나타낸다. 그룹은 특정 도메인에 소속되어야 하며 도메인 내에서만
고유한 이름을 갖는다.

### 2) Resoure
Resource 서비스는 프로젝트와 도메인에 대한 데이터를 제공한다.

#### a. 프로젝트(Projects)
프로젝트는 OpenStack에서 소유권의 기본 단위를 나타내며,
OpenStack의 모든 리소스는 특정 프로젝트에 소속되어야 한다.
프로젝트는 특정 도메인에 소속되어야 하며 도메인 내에서만 고유한 이름을 갖는다.
프로젝트의 도메인이 지정되지 않은 경우 기본 도메인에 추가된다.

#### b. 도메인(Domains)
도메인은 프로젝트, 사용자 및 그룹을 위한 고수준 컨테이너이다.
각 도메인은 정확히 하나의 도메인을 소유하며, API에서 볼 수 있는 이름 속성을 정의하는
네임스페이스를 제공한다. Keystone은 기본 도메인(Default)이라는 이름의 도메인을 제공한다.

Identity v3 API에서 속성의 고유성:

- 도메인 이름(Domain Name): 모든 도메인에서 글로벌로 고유함.
- 역할 이름(Role Name): 소속 도메인 내에서 고유함.
- 사용자 이름(User Name): 소속 도메인 내에서 고유함.
- 프로젝트 이름(Project Name): 소속 도메인 내에서 고유함.
- 그룹 이름(Group Name): 소속 도메인 내에서 고유함.


도메인의 컨테이너 아키텍처 덕분에 OpenStack 리소스 관리를 위임하는 방법으로 사용할 수 있다.
한 도메인의 사용자는 적절한 권한이 부여되면 다른 도메인의 리소스에도 접근할 수 있다.

### 3) Assignment
Assignment 서비스는 역할 및 역할 할당에 대한 데이터를 제공한다.

#### a. 역할(Roles)
역할은 최종 사용자가 얻을 수 있는 권한 수준을 정의하는 것이다. 역할은 도메인 또는 프로젝트 수준에서 부여될 수 있으며,
개별 사용자 또는 그룹 수준에서 할당 가능하다. 역할 이름은 소속 도메인 내에서 고유하다.

#### b. 역할 할당(Role Assignments)
Role, Resource, Identity를 포함하는 3-튜플 구조로 이루어져 있다.

### 4) Token
Token 서비스는 사용자의 자격 증명이 이미 확인된 후 요청 인증에 사용되는 토큰을 검증하고 관리한다.

### 5) Catalog
Catalog 서비스는 엔드포인트 탐색에 사용되는 엔드포인트 레지스트리를 제공한다.




# 참고문헌
- [오픈스택 - Keystone](https://docs.openstack.org/keystone/latest/)

