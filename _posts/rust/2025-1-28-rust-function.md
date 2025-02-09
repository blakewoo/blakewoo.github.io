---
title: Rust - 함수
author: blakewoo
date: 2025-1-28 16:00:00 +0900
categories: [Rust]
tags: [Rust] 
render_with_liquid: false
---

# Rust 문법

## 1. 함수
### 1) 표기법
러스트에서 함수는 관례적으로(필수는 아니다) 기본적으로 소문자와 언더바로
구성된 스네이크 케이스(snake case)를 사용한다.

```rust
fn main() {
    println!("Hello, world!");

    another_function();
}

fn another_function() {
    println!("Another function.");
}
```


### 2) 매개변수
함수에 입력되는 값으로 타입을 명시해줘야한다.

````rust
fn main() {
    another_function(5);
}

fn another_function(x: i32) {
    println!("The value of x is: {x}");
}
````

### 3) 구문과 표현식
rust에서는 두 가지 종류의 표현이있다.

#### a. 구문
어떤 동작을 수행하고 값을 반환하지 않는 명령
가령 함수의 정의나 변수에 대입하는 것 역시 구문에 해당한다.

예를 들어 아래와 같은 코드가 있다.
```rust
fn main() {
    let y = 6;
}

fn five() -> i32 {
    5
}
```

ley y = 6; 코드와 five 함수가 그런 이러한 구문의 예시다.


#### b. 표현식
결과값을 평가하는 것

아래와 같은 코드가 있다면
```rust
fn main() {
    let y = {
        let x = 3;
        x + 1
    };

    println!("The value of y is: {y}");
}
```
y안에 x+1 코드가 표현식이다.
x+1에 대해서 평가하여 혹은 연산하여 결과값을 반환한다.
이런 표현식의 경우 끝에 세미콜론을 붙이지 않는다

### 4) 반환값
함수는 반홥값이 있을수도 있고 없을수도 있다.
명시적으로 return 값으로 조기 반환을 할수도 있지만
함수 끝에 표현식을 두면 묵시적으로 반환하게 된다.
```
fn five() -> i32 {
    5
}

fn main() {
    let x = five();

    println!("The value of x is: {x}");
}
```


# 참고문헌
- [The Rust Programming Language - 함수](https://doc.rust-kr.org/ch03-03-how-functions-work.html)
