---
title: Rust - 제네릭 타입
author: blakewoo
date: 2025-2-22 20:00:00 +0900
categories: [Rust]
tags: [Rust] 
render_with_liquid: false
---

# Rust 문법

## 1. 제네릭 타입
제네릭을 사용하면 함수 시그니처나 구조체의 아이템에 다양한 구체적 데이터 타입을 사용할 수 있도록 정의할 수 있다.
총 4가지를 제네릭 데이터 타입으로 정의할 수 있다.

### 1) 함수
제네릭 함수를 정의할 때는, 함수 시그니처 내 매개변수와 반환 값의 데이터 타입 위치에 제네릭을 사용하면 된다.
제네릭 함수로 정의하면 데이터 타입마다 함수를 정의하지 않아도 되서 코드가 매우 깔끔해진다.   
아래와 같이 제네릭 함수를 정의할 수 있다.

```rust
fn mirror<T>(data: T) -> T {
    data
}

fn main() {
  println!("i32 data : {}", show(11));
  println!("f32 data : {}", show(1.1));
}
```

단, 데이터 타입마다 별도의 처리를 구현해야하는 경우가 있는데 이는 트레이트에 대해 설명할 때 추가적으로 업데이트하겠다.

### 2) 구조체
제네릭 구조체의 경우도 struct 명령어 옆에 원래라면 데이터 타입이 나올자리를 <T>를 통해 제네릭으로 표기해주면 된다.

```rust
struct Point<T> {
    x: T,
    y: T,
    z: T
}

fn main() {
    let integer = Point { x: 5, y: 10 , z: 30};
    let float = Point { x: 1.0, y: 4.0 , z: 3.3};
}
```

### 3) 열거형
이전에 Option 형과 같은 경우는 이전 열거형 파트에서 설명했다.

```rust
enum Option<T> {
    Some(T),
    None,
}
```

꼭 데이터 타입이 한 개가 아니라도 아래와 같이 사용할 수 있다.

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

### 4) 메서드
구조체에 정의된 self를 포함한 함수인 메서드도 반호나값과 데이터 타입에 대해서 제네릭 타입을 사용할 수 있다. 

```rust
struct Point<T> {
    x: T,
    y: T,
}

impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
    fn y(&self) -> &T {
        &self.y
    }
}

fn main() {
    let p = Point { x: 5, y: 10 };

    println!("p.x = {}", p.x());
    println!("p.y = {}", p.y());
}
```



# 참고문헌
- [The Rust Programming Language - 제네릭 데이터 타입](https://doc.rust-kr.org/ch10-01-syntax.html) 
