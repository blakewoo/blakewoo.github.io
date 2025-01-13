---
title: Devstack 간단한 운용
author: blakewoo
date: 2025-1-13 16:50:00 +0900
categories: [Trial and error]
tags: [Openstack, Devstack] 
render_with_liquid: false
---

# Devstack 간단한 운용

> ※ 설치 부분은 [이곳](https://blakewoo.github.io/posts/devstack-%EC%84%A4%EC%B9%98/) 에서 참조하길 바람 
{: .prompt-tip }

> ※ 본 포스팅은 2025년 1월 13일 기준으로 작성되었음
{: .prompt-tip }

## 목표
1. devstack에서 ubuntu 24.04를 대상으로 한 인스턴스를 생성한다.
2. 외부 환경에서 생성된 이미지에 엑세스한다.

## Horizon 접속
Devstack 설치시 접속했던 IP로 접속한다.
이후 devstack 설치시 지정했던 관리자 ID로 Login하여 Dashboard에 접속한다.

![img_2.png](/assets/blog/trial_error/devstack/simple_usage/img_2.png)

기본적으로 프로젝트 단위로 인스턴스가 운용된다.
따라서 작업할 프로젝트를 지정해주어야한다.
왼쪽 위에 보면 openstack 라벨 오른쪽에 프로젝트 선택할 수 있는 항목이 있는데
여기서 demo 프로젝트로 들어간다.

## 이미지 추가하기
ubuntu 24.04 버전 이미지를 추가해보도록 하겠다.   
먼저 ubuntu에서는 cloud에서 사용하는 img를 별도로 제공하고 있다.
[https://cloud-images.ubuntu.com/](https://cloud-images.ubuntu.com/) 에 들어가면 버전별로 명시가 되어있다.

![img.png](/assets/blog/trial_error/devstack/simple_usage/img.png)

이후 openstack Dashboard의 왼쪽 모록에 compute 항목에 이미지가 있다.   
해당 항목을 선택하면 기본적으로 cirros-0.6.3-x86_64-disk 이미지 하나만 떠있다.   
목록의 오른쪽 위에 이미지 생성을 누르면 아래와 같은 화면이 나온다.

![img_1.png](/assets/blog/trial_error/devstack/simple_usage/img_1.png)

- 이미지이름 : 편한대로 지정하면 된다.
- 이미지 소스 : 아까 ubuntu 홈페이지에서 받아온 img를 등록한다.
- 포맷 : "QCOW2 - QEMU 에뮬레이터 (Emulator)" 로 지정해준다

이후 오른쪽 아래에 이미지 생성 버튼을 눌러주면 이미지 생성이 완료된다.

## 보안 그룹 수정
별도로 새로 만들어도 되고 아니면 그냥 있는 default를 바꾸어도 된다.
어차피 테스트 용도이니 default에서 필요한 두 가지만 추가하도록 하겠다.

먼저 왼쪽 네트워크의 보안 그룹 메뉴에 들어간다 그리고 default 항목에 있는 규칙 관리
버튼을 클릭한다.

![img_9.png](/assets/blog/trial_error/devstack/simple_usage/img_9.png)

이후 ssh 통신에 관한 포트와 PING 체크를 위한 ICMP를 열어준다.

![img_10.png](/assets/blog/trial_error/devstack/simple_usage/img_10.png)


## Keypair 생성 
서버를 만들었을 때 대부분의 Cloud 서비스와 그러하듯
해당 서버로 접속하기 위해서는 Keypare가 필요하다. 
왼쪽 메뉴의 Compute 항목에 들어가서 키페어 메뉴를 선정하면 키페어를 생성할 수 있다.
이 메뉴에서 원하는 키 이름으로 원하는 타입으로 생성해줄 수 있다.
여기서는 키 이름을 test_key, 키 유형을 SSH 키로 지정하여 생성하여 다운로드 한다.

## 추가한 이미지로 서버 생성하기
왼쪽 메뉴의 Compute의 인스턴스로 들어간다.
이후 오른쪽 위에 있는 인스턴스 시작 버튼을 누른다.

![img_3.png](/assets/blog/trial_error/devstack/simple_usage/img_3.png)

위와 같이 원하는대로 인스턴스 이름을 지정하고 가용구역과 개수는 그대로 두고 Next를 누른다.

![img_4.png](/assets/blog/trial_error/devstack/simple_usage/img_4.png)

어떤 이미지로 인스턴스를 생성할 것인가에 대한 부분이다.
부팅 소스는 이미지로 그대로 두고 볼륨은 넉넉하게 30GB로 지정한다.
사용 이미지는 아까 받아서 이미지로 만들어둔 ubuntu 24.04의
오른쪽에 있는 위 화살표 버튼을 눌러서 지정한다.
여기서는 테스트기 때문에 인스턴스 삭제시 볼륨 삭제도 예로 설정해둔다.

![img_5.png](/assets/blog/trial_error/devstack/simple_usage/img_5.png)

해당 서버를 구동할 적절한 인스턴스 사이즈를 지정한다.
적절하게 m1.medium으로 지정해두었다.
Next를 누른다.

![img_6.png](/assets/blog/trial_error/devstack/simple_usage/img_6.png)

어느 네트워크에 서버를 둘 것인가에 대한 부분으로 어차피 route를 통해 엑세스할 수 있게 할 예정이다.
위와 같이 private에 둔다.

![img_7.png](/assets/blog/trial_error/devstack/simple_usage/img_7.png)

아까 만들어둔 keypare 역시 지정해준다.

## 제대로 생성되었는지 확인
인스턴스 리스트에서 인스턴스 이름을 클릭하면 아래와 같은 화면이 뜬다.

![img_8.png](/assets/blog/trial_error/devstack/simple_usage/img_8.png)

여기서 콘솔을 누르면 브라우저 내에서 콘솔에 엑세스 할 수 있다.
당장은 로그인하지 말고 그냥 콘솔이 뜨는지만 확인하면 된다.

## 유동 IP 할당하기
이대로는 접속할 수가 없다. 따라서 외부에서 접속할 수 있게 IP를 할당해주어야한다.
여기서 필요한게 유동 IP 할당하는 것이다.

![img_11.png](/assets/blog/trial_error/devstack/simple_usage/img_11.png)

해당 버튼을 누르면 아래의 창이 뜬다

![img_12.png](/assets/blog/trial_error/devstack/simple_usage/img_12.png)

+ 버튼을 눌러주면 아래의 창이 드는데 그냥 IP 할당 버튼을 눌러주면 자동을 할당된다.

![img_13.png](/assets/blog/trial_error/devstack/simple_usage/img_13.png)

이후 연결 버튼을 누르면 된다.

## Devstack 구동 중인 서버에서 확인하기

구동 중인 서버에서 terminal을 띄운 후 아래의 명령어를 입력한다.

```shell
ssh -i test_key.pem ubuntu@{유동 IP}
```

여기서 test_key.pem은 아까 생성한 test_key.pem 키를 사용하면 된다.

## 외부 접속 설정
그냥 서버 컴퓨터에서 접속이라면 아래와 같은 상황이다.

![img_14.png](/assets/blog/trial_error/devstack/simple_usage/img_14.png)

하지만 외부 접속은 아래와 같다.

![img_15.png](/assets/blog/trial_error/devstack/simple_usage/img_15.png)

사실 아키텍쳐상으로는 완전히 맞는 구성은 아니다.   
하지만 개념적으로 이해하기에는 충분하다고 보고 설명에 들어가겠다.

위 그림과 같이 ubuntu에서 해당 요청을 해당 port로 받았을 때 ip와 port를 매핑해주는게 필요하다.
이를 Forwarding 해주는게 필요하다.

아래와 같은 명령어로 rinetd를 설치해준다.
```shell
apt-get install rinetd
```

/etc/rinetd.conf에 위치한 rinetd 설정을 바꿔준다.
```
... ...
# bindadress bindport connectaddress connectport options...
0.0.0.0     8080       {지정된 유동 IP}     22
... ...
```
이후 rinetd를 다시 실행한다.

```shell
/etc/init.d/rinetd restart
```

이후 외부 컴퓨터에서 mapping한 정보로 접속하면 아래와 같이 접속 된다.

![img_16.png](/assets/blog/trial_error/devstack/simple_usage/img_16.png)



# 참고문헌
- [OPENSTACK - DEVSTACK](https://docs.openstack.org/devstack/latest/index.html)
