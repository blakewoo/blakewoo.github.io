---
title: OPENSTACK - Neutron
author: blakewoo
date: 2025-1-10 15:00:00 +0900
categories: [Openstack]
tags: [Openstack, neutron] 
render_with_liquid: false
---

# neutron
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




# 참고문헌
- [오픈스택 - neutron](https://docs.openstack.org/neutron/latest/)
- [Somaz의 IT 공부 일지 - Openstack Neutron이란?](https://somaz.tistory.com/123)
