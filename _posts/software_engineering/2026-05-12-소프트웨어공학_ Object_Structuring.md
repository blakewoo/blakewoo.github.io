---
title: 소프트웨어 공학 - OOLC - Object Structuring (객체 구조화)
author: blakewoo
date: 2026-5-14 23:00:00 +0900
categories: [software engineering]
tags: [software engineering] 
render_with_liquid: false
use_math: true
---

# Object Structuring
객체 구조화는 아래의 순서를 따른다.

## 1. Client/Server 식별
### 1) Client/Server 서브시스템 구조
모든 객체는 반드시 아래의 스테레오타입 중 하나를 명시하며 하나의 객체가 두 가지 역할을 겸하지 않도록 단일 책임을 유지한다

<table>
    <tr>
        <td>유형</td>
        <td>스테레오 타입</td>
        <td>역할 </td>
    </tr>
    <tr>
        <td>Interface Object</td>
        <td>«user interface»</td>
        <td>외부 사용자(Actor)와 시스템 간 통신 — 사용자 입출력 처리 </td>
    </tr>
    <tr>
        <td>Interface Object</td>
        <td>«input/output device interface»</td>
        <td>입출력 겸용 장치와 시스템 간 통신 </td>
    </tr>
    <tr>
        <td>Interface Object</td>
        <td>«output device interface»</td>
        <td>출력 전용 장치와 시스템 간 통신 </td>
    </tr>
    <tr>
        <td>Entity Object</td>
        <td>«entity»</td>
        <td>데이터 저장 및 관리 — 도메인 모델 클래스 </td>
    </tr>
    <tr>
        <td>Control Object</td>
        <td>«state dependent control»</td>
        <td>상태에 따라 행동이 달라지는 제어 흐름 </td>
    </tr>
    <tr>
        <td>Control Object</td>
        <td>«control»</td>
        <td>상태 독립적 실행 순서 제어 — 흐름 제어 클래스 </td>
    </tr>
    <tr>
        <td>Business Logic Object</td>
        <td>«business logic» / «transaction manager»</td>
        <td>서버 측 비즈니스 규칙 처리 — 서비스·트랜잭션 클래스 </td>
    </tr>
</table>

### 2) UC 패키징
- 각 Concrete Use Case는 Client UC와 Server UC의 쌍으로 패키징한다
- Client UC와 Server UC는 «include» 관계로 연결하고, Client UC가 Server UC를 호출한다
- Operator UC는 Client Subsystem 전용으로 패키징한다 
- Abstract Use Case도 Client Abstract UC + Server Abstract UC 쌍으로 패키징한다


## 2. Object 설계
### 1) Interface Object
- 외부 Actor·장치(External Class) 1개당 Interface Object 1개를 설계한다
- 장치 외부 클래스는 실제 수행하는 입출력 방향에 따라 스테레오타입을 결정한다
- 입출력 겸용: «input/output device interface» / 출력 전용: «output device interface»
- 외부 사용자(Actor) → «user interface» 스테레오타입 적용 - L1에서 식별된 모든 Actor를 빠짐없이 변환한다
- 각 시스템 인스턴스마다 Interface Object 인스턴스가 1개씩 존재한다

### 2) Entity Object
- 정보를 저장해야 하는 단계가 있으면 Entity Object를 식별한다
- 여러 클라이언트에서 공유해야 하는 Entity Object는 서버에 배치한다
- Static Model의 엔티티 클래스를 기반으로 서버 Entity Object를 결정한다
- 클라이언트에서 전송된 거래 데이터는 서버에서 Transaction Log로 영속 저장된다
- 일시적(Transient) 거래 데이터와 영속(Persistent) 거래 데이터를 분리하여 설계한다
  
### 3) Control Object
- 여러 단계의 실행 순서를 조율해야 할 때 Control Object를 도입한다
- 상태 독립적 제어는 «control» 스테레오타입을 사용한다
- Control Object는 직접 데이터를 저장하지 않는다 (Entity Object에 위임)
- 상호 배타적 활동(Mutex)은 하나의 Control Object로 통합 관리한다

### 4) 동시성 설계
- 여러 클라이언트가 동시에 접근하는 Entity Object는 상호 배제(Mutual Exclusion)를 보장한다
- 임계 영역의 범위는 첫 번째 검증(Validation) 시작부터 마지막 Entity 변경(Mutation) 완료까지로 한다
- 검증(Validation)과 변경(Mutation)을 별도 임계 영역으로 분리하지 않는다 (TOCTOU 취약점 방지)

### 5) Transient/Persistent 엔티티 분리
각 엔티티 데이터를 어디에 두고, 얼마나 오래 유지하는가를 결정하는 부분이다.
- 처리 세션 동안만 존재하고 서버에 영속 저장되지 않는 엔티티는 Transient Entity로 분류한다
- Transient Entity는 Client Subsystem에 배치한다
- Transient Entity는 메시지 시퀀스 단계별로 데이터를 누적하는 메서드를 가진다
- 처리 세션 완료 후, Transient Entity의 거래 데이터는 서버로 전송(transfer)되어 Transaction Log로 영구 저장(persist)된 다.

### 4) Business Logic object
- 서버에서 비즈니스 규칙을 처리하는 객체는 Business Logic Object로 설계한다
- 거래 유형(Transaction Type)마다 Transaction Manager를 1개씩 설계한다
- Business Logic Object(TransactionManager)는 Coordinator로부터 위임받은 요청을 처리하여 Entity Object를 조회·수정하고 응 답을 반환한다
- 클라이언트 요청의 단일 진입점으로 «coordinator» 객체를 설계한다 - coordinator는 거래 유형을 판단하여 적절한 TransactionManager에 위임 (캡슐화 원칙)


> ※ 추가 업데이트 및 검증 예정이다.
{: .prompt-tip }

# 참고 자료
- 서강대학교 박수용 교수님 강의자료 - 소프트웨어 공학

# 원문 참고 자료
- J. Rumbaugh et al, "Object-Oriented Modeling and Design", Prentice Hall, 1991
- I Jacobson et al, "Object-Oriented Software Engineering", Addison Wesley, Reading MA, 1992.
- H. Gomaa, “Chapter 6 - Designing Concurrent, Distributed, and Real-Time Applications with UML”, Addison Wesley Object Technology Series, July, 2000
