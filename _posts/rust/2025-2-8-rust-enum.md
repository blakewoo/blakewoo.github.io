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

if else 때와 같이 몇몇 케이스 외의 경우를 처리해야할 경우가 생길 수 있다.   
그런경우 other를 쓰거나 _로 케이스를 처리 할 수 있다.
```rust
fn main() {
  let dice_roll = 9;
  match dice_roll {
      3 => add_fancy_hat(),
      7 => remove_fancy_hat(),
      _ => reroll(),
  }

  fn add_fancy_hat() {}
  fn remove_fancy_hat() {}
  fn reroll() {}
}
```
혹은 reroll 대신에 ```()```를 입력하여 아무것도 안 할 수도 있다.


## 3. option 열거형
아래의 열거형은 표준 라이브러리에 정의된 열거형이다.

```rust
enum Option<T> {
    None,
    Some(T),
}
```

이는 이후 언급할 제네릭 타입의 자료형인데, Null Pointer Dereference(널 포인터 역참조) 처럼 Null로 리턴되는 값을 Null이 아닌 값처럼 사용할때
발생할 수 있는 문제점을 컴파일 타임에 방지하고자 만들어진 열거형 타입이다.

이 null이란 현재 어떠한 이유로 인해 유효하지 않거나, 존재하지 않는 하나의 값을 표기할때 사용하는데   
다른 언어에서 무분별한 null 사용으로 문제가 컸기에 rust에서는 null을 대체하기 위해서 만들었다.

이 option 열거형을 이용해서 2번의 match를 사용하는 예제는 아래와 같다.   
```rust
fn main() {
    fn plus_one(x: Option<i32>) -> Option<i32> {
        match x {
            None => None,
            Some(i) => Some(i + 1),
        }
    }

    let five = Some(5);
    let six = plus_one(five);
    let none = plus_one(None);
}
```

입력 받은 x가 Option<i32>라면 None일때 별도로 처리하고, Some일때 또 별도로 처리해준다.
None일 경우를 무조건 처리해주어야하기 때문에 값이 없을 때를 처리해줄 수 있는 것이다.   
개발자 입장에서는 번거로운 일이지만 컴파일 시점에 위험한 부분을 아는것이 차후 런타임에서 아는 것보다
결과적으로 효율적이기 때문에 다음과 같은 방법을 차용했다고 한다.

## 4. if let 흐름제어
입력값이 some 베리언트인 상황에서 if와 let을 이용하면 한 개만 처리하고 나머지는 무시해야하는
상황에서 간단하게 처리할 수 있다.
if와 let을 이용하지 않는다면 아래와 같이 구현해야한다.

```rust
fn main() {
    let config_max = Some(3u8);
    match config_max {
        Some(max) => println!("The maximum is configured to be {}", max),
        _ => (),
    }
}
```

하지만 if 와 let을 사용한다면 아래와 같이 간단하게 구현 가능하다.

```rust
fn main() {
    let config_max = Some(3u8);
    if let Some(max) = config_max {
        println!("The maximum is configured to be {}", max);
    }
}
```

물론 some 베리언트가 아님에도 if let은 아래과 같이 사용가능하다.
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

fn main() {
    let coin = Coin::Penny;
    let mut count = 0;
    if let Coin::Quarter(state) = coin {
        println!("State quarter from {:?}!", state);
    } else {
        count += 1;
    }
}
```
위의 경우에는 else를 포함하여 if에 체크된 케이스가 아닌 경우에도 처리할 수 있게 하였다.

# 참고문헌
- [The Rust Programming Language - 열거형 정의하기](https://doc.rust-kr.org/ch06-01-defining-an-enum.html) 
- [The Rust Programming Language - match 제어흐름 구조](https://doc.rust-kr.org/ch06-02-match.html)
- [The Rust Programming Language - if let을 이용한 간결한 제어흐름](https://doc.rust-kr.org/ch06-03-if-let.html)
