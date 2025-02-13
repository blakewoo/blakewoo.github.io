---
title: AWS firecracker 논문 해석
author: blakewoo
date: 2025-2-13 17:30:00 +0900
categories: [Paper]
tags: [Paper, AWS, Lambda, Firecracker] 
render_with_liquid: false
---


# Firecracker

> ※ 인터넷 및 논문을 읽고 분석했으나 해석이 잘 못되었을수도 있으므로 원문을 직접 살피는 것을 권고함
{: .prompt-tip }

## 1. 개요
Firecracker라고 하면 무엇인지 모르는 사람이 있을 것이다. 
하지만 "AWS Lambda"에 대해서 말하면 아는 사람은 꽤 될 것이다.    
AWS에서 나온 Lambda라는 서비스는 이 Firecracker가 적용된 서비스이다.   
이 Firecracker가 무엇인지 논문 제목을 이용해 설명하자면 서버리스 어플리케이션을 위한 경량 가상화가 되겠다.

aws의 개발자들이 만들었으며 현재 오픈소스로 운용되고 있다.
이에 관해 논문이 나와있는데, "Firecracker: Lightweight Virtualization for Serverless Applications"라는 제목이다.

이 논문에 대해서 읽고 어떤 내용이 담겨있는지 분석해볼 생각이다.

## 2. 논문

### 1) 개요
논문의 첫 부분에는 서버리스를 위한 가상화 솔루션이 필요하다고 적혀있다.   
전통적인 솔루션은 컨테이너와 가상화 둘 간에 고르게 되어있고 각각이 Trade-off가 있다는 것인데
가상화는 성능이 떨어지는 대신 보안이 뛰어나고, 컨테이너는 성능이 뛰어난 대신 보안이 떨어진다고 주장하고 있다. 그러면서
실제 서비스에서 성능과 보안 둘 중에 하나를 선택하는 것은 말도 안되므로 두 마리 토끼를 다 잡은 솔루션이 필요하다는 것이다.

이를 위해 서버리스에 특화된 가상머신 모니터(Virtual Machine Monitor, 이하 VMM)인 Firecracker를 개발했다고 한다.   
이 Firecracker는 성능과 보안성을 위하여 KVM을 통한 가상화를 유지한다.   
QEMU를 대체하여 운용되며 특정 부분의 요청을 Firecracker가 처리해줌으로써 성능의 향상과 적은 메모리를 사용하고 있다고 말한다.

Firecracker는 Unix API를 통해 제어되며 오버헤드가 5MB 이하이고, 어플리케이션이 부트되는데 125ms 이하의 시간이 들며 초당 150개의   
MicroVM을 만들 수 있다. 따라서 빠른 구동이 필요한 FaaS에 적합하다고 말한다.

Firecracker를 통해 만들어진 microVM은 몇몇 기능은 의도적으로 제공하지 않는데, 이는 필요한 부분만 만들어서 사용함으로써
빠른 구동에만 초점을 둔 탓이다.
따라서 Firecracker에서는 BIOS를 제공하지 않으며, 임의의 커널을 부팅할 수 없고, 레거시 장치나 PCI를 에뮬레이션하지 않으며,
VM 마이그레이션을 지원하지 않는다. 또한 Microsoft Windows를 부팅할 수 없다.

### 2) 구성
firecracker는 기본적으로 KVM을 통해 linux를 Hypervisor로 사용하며, QEMU를 대신하여 구동된다.
아래 그림은 논문과 공식 홈페이지, 공식 git의 문서를 참조하여 재구성한 그림이다.

![img.png](/assets/blog/paper/firecracker/img.png)

기본적으로 linux에서 돌아가되 firecracker에 포함된 API 관련 thread에서 모든걸 제어한다.
이 RESTful API Thread에서 VMM에 달려있는 성능 제한과 IMDS를 제어하며 microVM의 필요한 기능들을
VMM과 IMDS를 통해 제어한다.
Firecracker와 Guest OS가 구동중인 VM은 jailer를 통해 격리되며 이 jailer는 microVM이 뚫렸을 때를 대비한
2차 방어벽이다.

microVM당 하나의 Firecracker 프로세스가 구동된다. 이는 microVM을 관리하기 위함이며 해당 프로젝트안에는 이를
orchestration 하는 코드는 포함되어있지 않다. 
이는 AWS에서 사용할때 별도로 구현한거 같으며 이를 이용해서 containerd로 제어할 수 있는 별도의 프로젝트가 있다.

> ※ 본 포스팅은 추가적으로 업데이트 될 예정이다. 제대로 이해한게 맞는지 확인해야한다.
{: .prompt-tip }


# 참고문헌
- Alexandru Agache et al, "Firecracker: Lightweight Virtualization for Serverless Applications", the
  17th USENIX Symposium on Networked Systems Design  and Implementation (NSDI ’20) February 25–27, 2020
- [firecracker 공식 홈페이지](https://firecracker-microvm.github.io/)
- [firecracker 공식 github - design.md](https://github.com/firecracker-microvm/firecracker/blob/main/docs/design.md)
