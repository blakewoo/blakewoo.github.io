---
title: Rust - 열거형
author: blakewoo
date: 2025-2-8 23:00:00 +0900
categories: [Rust]
tags: [Rust] 
render_with_liquid: false
---

# Rust 문법

## 1. 열거형
열거형은 어떤 값이 여러 개의 가능한 값의 집합 중 하나라는 것을 나타내는 방법을 제공하는 방법이다.    
rust 뿐만 아니라 다른 언어들에서도 흔히들 제공하는 방식이다.

rust에서 열거형 정의 방법은 아래와 같다.
```rust
enum IpAddrKind {
    V4,
    V6,
}

fn main() {
    let four = IpAddrKind::V4;
    let six = IpAddrKind::V6;

    route(IpAddrKind::V4);
    route(IpAddrKind::V6);
}

fn route(ip_kind: IpAddrKind) {}
```

route 함수의 인자로 IpAddrKind를 쓰는 부분이다.
아예 어떤 형태인지 정의하고 싶다면 아래와 같이 정의할 수도 있다.

```rust
fn main() {
    enum IpAddr {
        V4(u8, u8, u8, u8),
        V6(String),
    }

    let home = IpAddr::V4(127, 0, 0, 1);

    let loopback = IpAddr::V6(String::from("::1"));
}
```

아니면 아예 아래와 같이 구조체를 지정할 수도 있다
```rust
#![allow(unused)]
fn main() {
struct Ipv4Addr {
    // --생략--
}

struct Ipv6Addr {
    // --생략--
}

enum IpAddr {
    V4(Ipv4Addr),
    V6(Ipv6Addr),
}
}
```

이는 열거형에는 어떤 형태의 데이터라도 넣을수 있기 때문이다.

## 2. match 제어흐름
다음 열거형에 따라 해당 값이 어떤 값인지 받고 그 값에 따라 다른 코드를 실행시킬 수 있다.   
아래의 예시 코드를 보자.

```rust
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter,
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => {
            println!("Lucky penny!");
            1
        }
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter => 25,
    }
}

fn main() {}
```

match 식별자 이후 값을 비교하고 싶은 변수 값을 넣고 중괄호안에는 해당 값이
어떤 타입인지에 따라 반환 값이 다르다.   
짧게 쓰려면 중괄호를 생략하고 콤마로 구분할 수 있지만 2줄 이상 쓰기 위해서는 중괄호로 묶어야한다.

혹은 열거형 안에 열거형을 통해서 match를 처리할 수도 있다.   
아래의 코드를 보자.

```rust
#[derive(Debug)]
enum UsState {
    Alabama,
    Alaska,
    // --생략--
}

enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(UsState),
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("State quarter from {:?}!", state);
            25
        }
    }
}

fn main() {
    value_in_cents(Coin::Quarter(UsState::Alaska));
}
```

미국의 각 주마다 Quarter는 모양이 다르다고 한다.   
이런 상황을 코드로 표현하고자 할 때 Coin의 Quarter는 UsState, 즉 미국의 주에 대한 정보를 넣음으로써
별도로 처리해 줄 수 있다. 위 코드는 그 부분을 추가한 코드이다.

> ※ 본 포스팅은 추가적으로 업데이트 될 예정이다.
{: .prompt-tip }

# 참고문헌
- [The Rust Programming Language - 열거형 정의하기](https://doc.rust-kr.org/ch06-01-defining-an-enum.html) 
- [The Rust Programming Language - match 제어흐름 구조](https://doc.rust-kr.org/ch06-02-match.html)
