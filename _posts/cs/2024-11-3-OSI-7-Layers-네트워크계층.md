---
title: OSI 7 Layer - 네트워크 계층
author: blakewoo
date: 2024-11-3 22:00:00 +0900
categories: [Computer science]
tags: [Computer science, Network] 
render_with_liquid: false
use_math: true
---

# OSI 7 Layer - 네트워크 계층
## 1. 개요 
이전에 Switch를 이용해서 다중 통신 환경을 구성했었는데, 이런 환경을 근거리 통신망(Local Area Network, LAN)이라고 한다.   
이렇게 구성된 근거리 통신망끼리 통신하기 위해서는 어떻게 하면 될까?
이럴때를 위해서 필요한게 IP 주소이다.

## 2. IP 주소의 구성
IP 주소는 현재 총 2가지 버전이 있다.

### IPv4
총 12자리이며, 4부분으로 끊어진다.

```
XXX.XXX.XXX.XXX
```

XXX 부분은 0~255까지 총 4개로 이루어져있다.   
그 이유가 각 부분마다 32bit로 이루어져있기 때문이다.   
IPv4의 최대 개수는 이론상 4,294,967,296개이다.

### IPv6
총 128bit로 이루어져있으며 16비트씩 8자리로 이루어져있다.    
각 부분은 콜론(:)으로 구분된다.
```
XXXX:XXXX:XXXX:XXXX:XXXX:XXXX:XXXX:XXXX
```
IPv4가 0~255까지인 반면 IPv6는 0000~FFFF(16진수)까지의 값을 가질 수 있다.
따라서 이론상 $2^{128}$개의 주소를 가질 수 있다.

## 3. IP 주소 통신 방식
IP주소를 이용하여 네트워크를 구성하게 되면 통신 대상 컴퓨터를 빠르게 찾을 수 있다.   
이는 Subnet을 이용한 네트워크 쪼개기를 통해 쓸데 없는 라우팅을 줄일 수 있기 때문이다.

※추가 포스팅 예정

## 4. DNS
IP 주소 같은 숫자를 사람이 외우기에는 꽤나 헷갈린다.   
따라서 Domain이라는 것을 도입했는데 이 Domain이라는 것은 일종의 사람이 기억하기 쉬운 문자열 형태이다.   
사용자가 Domain 주소로 요청을 하면 도메인 네임 서버라는 곳에서 해당 도메인의 IP 주소가 무엇인지 확인하여
반환해주고 사용자는 그 IP주소로 접근을 하게 된다.

※추가 포스팅 예정


# 참고 자료
- [위키백과 - OSI 모형](https://ko.wikipedia.org/wiki/OSI_%EB%AA%A8%ED%98%95)
- [위키백과 - IPv4](https://ko.wikipedia.org/wiki/IPv4)
- [위키백과 - IPv6](https://ko.wikipedia.org/wiki/IPv6)
