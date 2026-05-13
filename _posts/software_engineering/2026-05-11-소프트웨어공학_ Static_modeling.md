---
title: 소프트웨어 공학 - OOLC - Static modeling (정적 모델링)
author: blakewoo
date: 2026-5-14 23:00:00 +0900
categories: [software engineering]
tags: [software engineering] 
render_with_liquid: false
use_math: true
---

# Static modeling
정적 모델링은 아래의 절차를 따른다.

## 1. Problem Domain Static Modeling
물리적인 클래스를 식별하는 과정이다. 말 그대로 물리적 클래스 목록을 작성하며 도메인 용어를 확정하는 단계이다.

여기서 식별된 물리적 클래스는 Entity Class Static Modeling에서 소프트웨어 엔티티 클래스로 매핑   
매핑은 물리적클래스 → 엔티티 클래스의 1:1 또는 1:N로 변환한다.

이때 클래스 명을 지을때 소프트웨어 설계 용어나 아키텍처 용어나 설계 패턴 용어나 계층 용어 같은것을 사용하지 않고
명확한 물리적인 이름을 지어야한다.

예를 들어 ATM에 대해서 물리적인 클래스를 식별하면 클래스 이름이 ATM이어야하지, ATMController과 같은 다른 용어를 쓰면 안된다.   
이는 그 다음 절차를 다룰때 용어의 혼동이 오기 때문에 지양해야한다.

### 1) 클래스간의 관계
가령 ATM과 Bank와 CardReader라는 물리클래스가 있다고 해보자. 이를 예시로 클래스간의 관계는 아래와 같이 세 분류로 나눌 수 있다.

• Aggregation (ATM은 Bank 없이도 물리적으로 존재가 가능하다)   
• Composition (CardReader는 ATM의 일부, ATM 없이 독립적으로 존재가 불가하다)   
• Association (구조적 포함이 아닌 단순 연관, 예를 들면 대학과 학과)

은행 시스템에서 물리적 클래스와 관계, 다중성을 포함한 예시는 아래와 같다.

<table>
    <tr>
        <td>유형</td>
        <td>표기</td>
        <td>Bank system에서 예시</td>
    </tr>
    <tr>
        <td>1:1</td>
        <td>1──1 </td>
        <td>ATM ──Has──&gt;  CardReader</td>
    </tr>
    <tr>
        <td>수치 지정</td>
        <td>1,2</td>
        <td>Account ──Modifies──&gt; ATMTransaction</td>
    </tr>
    <tr>
        <td>0 .. 1, 선택적</td>
        <td>0..1</td>
        <td>Customer ──Owns──&gt; DebitCard</td>
    </tr>
    <tr>
        <td>N:M</td>
        <td>*──*</td>
        <td>Customer ──Owns──&gt; Account</td>
    </tr>
    <tr>
        <td>1:N</td>
        <td>1──1..*</td>
        <td>Bank (1) ──Has──────────&gt; (1..*) ATM</td>
    </tr>
</table>


## 2. System Context Static Modeling
"무엇이 시스템 안에 있고, 무엇이 시스템 밖에 있는가"를 결정하는 단계이다.

시스템 경계를 정의하고 외부 클래스(External Class)를 식별한다.
외부 클래스는  Object Structuring에서 Interface Object로 1:1 변환된다.

### 1) 외부 클래스 식별
시스템을 하나의 블랙박스(aggregate class)로 볼 때, 그 경계 밖에서 시스템에 입력을 주거나 시스템으로부터 출력을 받는 존재로 
시스템이 직접 구현·제어하는 대상이 아니라, 시스템이 인터페이스(interface)해야 하는 대상이다.

### 2) Class 스테레오 타입
클래스의 각 스테레오 타입의 예시는 아래와 같다.

- «system»   
  시스템 전체를 하나의 Aggregate Class로 표현

- «entity»   
  데이터 집약적·영속적 클래스. DB 매핑 대상

- «subsystem»   
  시스템의 부분 서브시스템

- «external input device»    
  시스템에 입력 제공 물리 장치
  
- «external output device»   
  시스템이 출력 보내는 장치

- «external system»   
  데이터 주고받는 외부 시스템
  
- «external user»    
  표준 I/O로 상호작용하는 인간

- «external I/O device»   
  입출력 모두 수행

- «external Timer»   
  시간을 파악해야하거나 외부 타이머 이벤트가 필요한 경우 사용

## 3. Entity Class Static Modeling

### 1) 물리적 클래스를 엔티티 클래스로 변환한다.
- 하나의 물리적 클래스가 복수의 논리적 개념을 포함하는 경우 각각 별도의 엔티티 클래스로 분리한다
- External Class로 분류된 클래스는 엔티티 클래스로 변환하지 않는다
- 중복 정보를 담는 클래스는 별도 엔티티로 만들지 않고 기존 클래스를 재활용한다
- 모든 엔티티 클래스는 «entity» 스테레오타입을 명시한다

### 2) 엔티티 클래스 구조
#### a. 계층 구조
각 엔티티 클래스는 아래와 같은 조건을 기반으로 계층을 확립한다.

- 복수 유형이 공통 속성을 공유하는가 -> Superclass 적용
- 각 유형이 고유 속성을 별도 보유하는가 -> Subclass 적용
- 두 조건 모두 충족 시 계층 구조 적용
- 클래스 간에 다중성은 명시하되 서브 클래스의 경우 생략이 가능함.

#### b. 연관 클래스 (Association Class)
- 두 엔티티 클래스 사이의 연관에 속성이 필요하면 Association Class로 독립 모델링한다
- Association Class는 코드에서 독립된 엔티티 클래스 파일로 생성한다
- Association Class의 속성은 연관 양쪽 클래스의 식별자(FK)와 관계 고유 속성을 포함한다
  
### 3) 클래스 속성(Attribute) 명세
- 속성 표기 형식
  - 모든 엔티티 클래스의 속성은  속성명 : 타입  형식으로 명시한다
  - 속성의 가시성은 + (public) / - (private) / # (protected)로 표기하며, 엔티티 속성은 기본적으로 - (private)으로 정의한다
  - 기본값(Default Value)이 있는 속성은  = 값 형식으로 명시한다
  - 컬렉션 속성은  속성명 : Collection<타입>  형식으로 명시하고, 순서가 있는 경우 List, 키-값 쌍인 경우 Map으로 구분한다
  - Superclass에 정의된 속성은 Subclass에 중복 정의하지 않는다  


- 데이터 타입 표준
  - 속성에 각각 데이터 타입을 정의하는 것이다. 이 때 사용금지 타입도 명시한다.  
  - 유한한 값의 집합을 가지는 속성은 enum 타입으로 별도 정의하되, 타입이 결정되지 않은 속성을 명세에 남기지 않는다
  

- 열거형(Enum) 정의
  - enum 타입으로 정의된 속성은 별도 «enumeration» 클래스로 명시한다
  - 열거 상수는 대문자 스네이크 케이스(UPPER_SNAKE_CASE)로 표기한다
  - 열거형 클래스는 어느 엔티티 클래스에서 사용하는지 의존 관계(«use»)로 표시한다

> ※ 추가 업데이트 및 검증 예정이다.
{: .prompt-tip }

# 참고 자료
- 서강대학교 박수용 교수님 강의자료 - 소프트웨어 공학

# 원문 참고 자료
- J. Rumbaugh et al, "Object-Oriented Modeling and Design", Prentice Hall, 1991
- I Jacobson et al, "Object-Oriented Software Engineering", Addison Wesley, Reading MA, 1992.
- H. Gomaa, “Chapter 6 - Designing Concurrent, Distributed, and Real-Time Applications with UML”, Addison Wesley Object Technology Series, July, 2000
