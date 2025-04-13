---
title: 파일 시스템 - Object Storage System
author: blakewoo
date: 2025-4-13 14:30:00 +0900
categories: [File system]
tags: [File system, Object Storage, Hardware]
render_with_liquid: false
---

# Object Storage System

## 1. 도입
1956년 IBM에서 등장한 최초의 HDD인 RAMAC 이후로 block device는 근 70여년간   
컴퓨터의 보조 기억장치로써 사용되어왔고, 이후 등장한 SSD 역시 block device로써 작동했으며   
이후 등장한 대부분의 File system들은 이 block device에 맞춰져서 개발되었다.   

그러던 2005년에 IBM Journal에서 새로운 개념이 등장했다.   
바로 Object-based Storage 이다.

## 2. Object
Object-based Storage에 대해서 알려면 먼저 Object가 무엇인지부터 알아야한다.   
Object란 데이터와 그에 연관된 속성, 메타데이터까지 포함한 형태의 가상 컨테이너라고 말할 수 있다.

이 Object란 파일일수도, 테이블일 수도, 혹은 여러 파일의 종합일수도 있으며 심지어 같은 Object를   
포함할 수 도 있다. 또한 고정된 크기였던 block과는 달리 Object의 크기는 가변적이다.

Object-based Storage란 block이 아닌 위와 같은 Object를 파일 입출력의 단위로 사용하는
스토리지 시스템이라고 보면 된다.

## 3. Object-based Storage background
Object-based Storage가 등장하게된 배경은 아래와 같다.

### 1) Disk 성능의 한계
이때 한창 HDD 성능 향상에 한계를 느끼고 있을 때였다.   
Platter 속도를 높이는 것도 한계가 있었고, 미리 읽어오는 Read-ahead 방식이라던지,
disk queue scheduling과 같은 방식을 써도 큰 효과를 보지 못했을 때이다.   
여기서 말하기를 block device에서 block 형태로 엑세스하는 것에는 한계가 왔다고 말하고 있다.

또한 데이터간의 관계(Relation)이 있지만 block device는 이러한 관계를 제대로 활용하지 못하여
성능 향상에 써먹을 수 없다고 한다.

```
물론 이 문제가 제기 되었을 당시에 2005년이었기에 등장했던 내용으로 SATA 기반 SSD가 등장(2007년)하기
전이었다. 그 이전에도 SSD를 차용한 방식은 좀 있었던 것 같지만 HDD가 주력인 시점에서 나왔기 때문에
한계에 봉착했다고 이야기하는 것 같다.
```

### 2) 보안 위협
어떤 파일에 대해서 엑세스하는데 있어서 block device는 제한이 없다. 관리자 권한이 있다면
누구든 어느 block에 쓰고 읽을 수 있으며 이는 무결성 문제와 보안 문제를 야기하기 때문이다.

## 4. Object Storage Drives (OSD)
객체 단위로 데이터를 저장하는 데이터 저장 아키텍처를 Object Storage Drives (이하 OSD)라고 한다.   
이를 위한 명령어 셋이 필요했고, 이에 따라 ANSI 2004년에 표준이 생겼다.

아래의 그림을 보자

![img.png](/assets/blog/file_system/OSD/img.png)

위 그림은 2005년에 IBM에서 나온 Journal에 수록된 그림이다.
전통적인 스토리지 구조에 비해 File system에 관한 부분이 Storage device에도
포함이 되어있는 것을 알수 있다. 이는 스토리지가 단순 저장뿐만 아닌 다수의 DMA와
내부 스케줄링을 위한 CPU를 가지게 되었기 때문에 운용 가능하며 IBM에서도 해당 부분을
염두에 두고 제시한게 아닌가 싶다.


> ※ 내용 업데이트 및 추가적인 검증 예정
{: .prompt-tip }

# 출처
- 서강대학교 김영재 교수님 강의자료 - 고급 데이터베이스
- IBM Journal 2005
