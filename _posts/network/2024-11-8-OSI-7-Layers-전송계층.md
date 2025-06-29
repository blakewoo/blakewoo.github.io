---
title: OSI 7 Layer - 전송 계층과 TCP
author: blakewoo
date: 2024-11-10 19:00:00 +0900
categories: [Network]
tags: [Network, OSI 7 Layer]
render_with_liquid: false
use_math: true
---

# OSI 7 Layer - 전송 계층과 TCP
## 1. 개요 
네트워크 계층까지는 어떻게 다수와 통신할 것인가에 대한 내용이었다면 전송 계층에서는
어떻게 이 통신간의 신뢰성과 다중 연결을 유지하는 방법에 대한 내용이다.

신뢰성이란 내가 지금 받고 있는 데이터가 옳은가?에 대한 내용이라고 볼 수 있겠다.   
가령 전송 과정에서 노이즈로 인해 데이터가 변경된건 아닌지? 아니면 누군가의 해킹으로 인해 데이터가 변조된건 아닌지?
혹은 큰 데이터의 경우 나눠서 전송하게 되는데 이렇게 나눠져서 받고 있는 데이터가 순서대로 오고 있는건 맞는지 등에 대한 내용이다.

다중 연결이란 한 개의 컴퓨터에서 다수의 컴퓨터를 상대로 연결을 하는 것을 말한다.   
우리 실생활에서도 자주 찾아볼 수 있는데, 웹 서핑을 하면서 백신 업데이트를 하는 경우를 생각해볼 수 있다.   
웹 서버로 요청하고 응답받는 연결이 있고, 백신 업데이트 서버에 요청하고 응답받는 연결이 있는 것이다.   
하나가 끊어지고 하나가 다시 연결되는 형태가 아닌 동시에 연결하고 있는 것이다.   
이렇게 연결에 대한 구분자는 미리 정의된 번호로 구분하게 되는데 이를 Port 번호라고 한다.

## 2. 신뢰성을 유지하는 법
이전에는 컴퓨터와 컴퓨터를 직접 연결하는 회로 교환 방식이었다. 하지만 이렇게 통신하니 해당 회선을 사용중일때
다른 컴퓨터가 대상 회선을 사용할 수 없는 문제가 생겼다. 그래서 회선은 공유하되 연결은 계속 유지할 수 있는 방법으로
데이터를 쪼개서 전달하는 방식인 패킷 교환 방식(Packet Switching)으로 사용하게 되었다.

패킷 교환 방식으로 사용하게되면 아무래도 회선을 직접 연결하는 것과 다른 신뢰성을 보장하는 방법을 강구해야한다.   
때문에 보내려는 데이터 앞에 목적지와 쪼개진 패킷 중에 몇번째 패킷인지를 명시하게 되는데 이를 캡슐화라고 한다.   
응용 계층에서 데이터를 보내며 계층을 하나씩 지날때마다 헤더가 붙어서 캡슐안에 넣는 것과 같다고 해서 캡슐화라고 한다.

캡슐화를 하는 과정에서 전송계층에서 헤더를 붙이는 방법은 두 가지가 있다.

### 1) TCP (Transmission Control Protocol)
신뢰성을 보장하는 통신방법이다. TCP 헤더는 아래와 같다.

![img.png](/assets/blog/cs/network/osi_7_layer_transmission/img.png)

- Source/Destination Port Number : 송수신측 포트 번호
- Sequence Number : 일련번호, 바이트 단위로 구분되어 순서화 되는 번호이다.
- Acknowledgement Number : 확인 응답 번호
- HLEN : 헤더 길이 필드(Header length)
- URG, ACK, PSH, RST, SYN, FIN : 데이터 제어 관리 플래그(3-way handshake, 송신 종료 알림)
- Window size : 세그먼트의 크기
- Checksum : 헤더 데이터에 문제가 없는지 확인하는 체크섬
- Urgent pointer : TCP 세그먼트에 초함된 긴급 데이터의 마지막 바이트에 대한 일련 번호
- Option and Padding : TCP 관련 옵션을 최대 40바이트까지 설정 가능

#### a. 일련번호와 확인 응답 번호 (Sequence Number, Acknowledgement Number)
데이터의 시작과 끝을 구분하기 위해 있는 것으로 송신측과 수신측에서 각기 쓰이는 의미가 다르다.

- 송신측   
  일렵 번호 : 현재 보내는 데이터의 시작 번호     
  확인 응답 번호 : 전체 데이터의 끝 번호    

- 수신측    
  일련 번호 : 전체 데이터의 끝 번호     
  확인 응답 번호 : 현재까지 받은 데이터의 끝 번호

이러한 일련번호와 확인 응답 번호는 아래의 윈도우 사이즈와 같이 쓰인다.

#### b. 윈도우 사이즈와 슬라이딩 윈도우
한번에 전송하는 데이터의 단위를 세그먼트(Segment)라고 한다.
여기서 말하는 윈도우 크기는 세그먼트를 일시적으로 저장하는 버퍼의 한계 크기이다.   

이 세그먼트가 너무 작으면 송수신측에서 너무 빈번하게 데이터를 보내야한다.
세그먼트가 너무 크면 수신측에서 데이터를 받기 어려울 수 있다.
그렇기 때문에 실질적인 전송 이전에 수신측의 윈도우 크기에 송신측의 윈도우 크기를 맞추어 설정한다.

윈도우 크기를 맞춰서 조정한 뒤 한번에 처리할 수 있는 양을 정해두고 아래와 같이 움직인다.

![img.png](/assets/blog/cs/network/osi_7_layer_transmission/img_9.png)

송신측에서 1 ~ 4까지 전송 준비가 되어있음 (빨간 색 네모가 윈도우)

![img_1.png](/assets/blog/cs/network/osi_7_layer_transmission/img_10.png)

데이터 1,2를 전송하고 3,4는 아직 전송하지 않았음

![img_2.png](/assets/blog/cs/network/osi_7_layer_transmission/img_11.png)

전송한 데이터 1,2에 대해서 ACK를 받았고 ACK 된 만큼 윈도우를 이동

위와 같이 한번에 처리하는 방식을 슬라이딩 윈도우 방식이라고 한다.


#### c. TCP 연결 시작 (3-way handshake)
송신측에서 데이터를 보내기전에 수신측이 준비가 되었는지 확인하는 과정이다.

절차는 아래와 같다.

1. 송신측에서 수신측으로 SYN을 보낸다.   
![img_2.png](/assets/blog/cs/network/osi_7_layer_transmission/img_2.png)

2. 수신측에서 송신측으로 SYN-ACK을 보낸다.    
![img_3.png](/assets/blog/cs/network/osi_7_layer_transmission/img_3.png)

3. 송신측에서 수신측으로 ACK를 보내고 데이터를 전송한다.    
![img_4.png](/assets/blog/cs/network/osi_7_layer_transmission/img_4.png)

#### d. TCP 연결 종료

1. 송신 측에서 수신 측으로 FIN을 보낸다.      
![img_5.png](/assets/blog/cs/network/osi_7_layer_transmission/img_5.png)

2. 수신 측에서 송신 측으로 ACK을 보낸다.    
![img_6.png](/assets/blog/cs/network/osi_7_layer_transmission/img_6.png)

3. 송신 측에서 수신 측으로 FIN을 보낸다.    
![img_7.png](/assets/blog/cs/network/osi_7_layer_transmission/img_7.png)

4. 수신 측에서 수신 측으로 ACK을 보낸다.    
![img_8.png](/assets/blog/cs/network/osi_7_layer_transmission/img_8.png)

### 2) UDP (User Datagram Protocol)

![img_1.png](/assets/blog/cs/network/osi_7_layer_transmission/img_1.png)

- Source Port/Destination Port : 송수신 포트
- Length : 패킷 전체 길이
- Checksum : 패킷 전체 체크섬

UDP의 경우 TCP와는 달린 신뢰성이 낮다. 이는 위의 헤더만 봐도 알 수 있는데
메세지가 제대로 도착했는지 확인하지 않으며, 수신된 데이터의 순서를 맞추지 않고 흐름제어 역시 없다.
Checksum을 제외한 오류제어 기능 역시 없다.

대체 이런 통신 방식을 어디에 쓰는가 싶지만 이는 확인 및 흐름제어같은 기능이 빠져있기 때문에
TCP에 비해 굉장히 빠르다. 그 때문에 빠른 요청과 응답이 필요한 실시간 응용에 적합하며
일 대 다 전송에 좋다. 따라서 미디어 스트리밍과 같은 것에 많이 사용된다.

## 3. 다중 연결을 유지하는 법
한 개의 컴퓨터에 다수의 연결을 유지하기 위해서는 개요에서 설명했던 것과 같이 Port를 사용한다.

### 1) Port
포트 번호는 0에서 65535까지 사용할 수 있다.
이는 포트 번호 지정용으로 2Bytes를 사용하기 때문이다.
0~1023 까지는 예약된 포트번호로 이미 지정된 번호가 있다. 
이를 well-known port라고 한다. 그 예시로는 아래와 같다.

- SSH: 22
- SMTP: 25
- DNS: 53
- HTTP: 80
- POP3: 110
- HTTPS: 443

# 참고 자료
- [위키백과 - OSI 모형](https://ko.wikipedia.org/wiki/OSI_%EB%AA%A8%ED%98%95)
- [정보통신기술용어해설 - TCP Header   Transmission Control Protocol Header   TCP 헤더](http://www.ktword.co.kr/test/view/view.php?no=1889)
- [변계사 Sam의 테크 스타트업! - 쉽게 이해하는 네트워크 11. 인터넷의 TCP/IP 프로토콜과 패킷 교환 방식](https://better-together.tistory.com/110)
- [제이크서 위키 블로그 - OSI 4계층 전송 계층 (Transport Layer) 알아보기](https://jake-seo-dev.tistory.com/401)
