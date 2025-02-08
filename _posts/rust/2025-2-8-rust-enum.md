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

> ※ 포스팅 추가 예정
{: .prompt-tip }



# 참고문헌
- [The Rust Programming Language - 열거형 정의하기](https://doc.rust-kr.org/ch06-01-defining-an-enum.html) 
