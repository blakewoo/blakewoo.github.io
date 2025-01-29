---
title: Rust - 함수
author: blakewoo
date: 2025-1-28 23:00:00 +0900
categories: [Rust]
tags: [Rust] 
render_with_liquid: false
---

# Rust 문법

> ※ 본 포스팅은 추가적으로 업데이트 될 예정이다.
{: .prompt-tip }

## 1. 주석
당연하지만 rust도 코드에 포함되지 않고 부연설명을 달 수 있는 주석 기능이 있다.
한 줄 혹은 다수의 줄을 주석으로 입력할 수 있다.

아래는 한줄 주석의 예시이다.
```rust
// 원하는 내용 앞에 두개의 슬래시를 입력하면 한 줄 주석처리가 된다.
```

두개 슬래시로 주석처리하기 많은 양의 주석 같은 경우 아래와 같이 처리할 수 있다.
```rust
/* 다수 줄의 주석은 다음과 같이 입력 할 수 있다.
   많은 양의 내용을 주석처리할 때 유용하다. */
```

## 2. 흐름 제어
조건에 따라 특정 코드를 실행하고 조건이 만족하는 동안 특정 코드를 반복수행하는 것을 말한다.
조건문이나 반복문이 이에 해당한다.

### 1) 조건문
#### a. if 표현식
조건에 따라 특정 코드를 실행 시킬 수도 있고 실행하지 않을 수도 있다.

예를 들어보자.
```
fn main() {
    let check_number = 1;

    if check_number < 2 {
        println!("check_numver가 2보다 작다");
    }
}
```

if 식별자 바로 뒤에 공백 이후에 조건이 나타난다. 그리고 중괄호로 실행대상이 되는 코드를 감싼다. 
여기서 조건문은 bool 타입으로 나타하는 형태여야한다.
bool 외의 다른 값을 반환하는 표현식이라면 에러가 난다.
다른 언어들이 그러하듯 if 안에 조건문이 만족하지 않는 경우도 추가 할 수 있다.
```rust
fn main() {
    let check_number = 1;

    if check_number < 2 {
        println!("check_numver가 2보다 작다");
    } else if check_number == 2 {
        println!("check_numver가 2보다 같다");
    } else {
        println!("check_numver가 2보다 크다");
    }
}
```

if와 else if는 조건문이 포함되어야하지만 else는 포함되지 않아도 된다.

변수에 값을 할당하기 위해 if 구문을 사용할 수도 있는데 다른 언어에서 삼항 연산자와 같은 역할을 할 수 있다.
```rust
fn main() {
    let check = true;
    let number = if check { 5 } else { 6 };

    println!("{number}");
}
```

### 2) 반복문

#### a. loop
다른 언어들에서 while 문이나 for의 조건문 부분에 true를 주는 식으로 무한 루프를 구현하지만
rust에서는 아예 loop 문을 통해 무한 루프를 구현할 수 있다.

다른 언어들과 마찬가지로 continue와 break 문을 통해
아래 코드를 건너 뛰고 루프문 처음부분으로 돌아가거나 루프를 탈출하거나 할수 있다.
```rust
fn main() {
    let mut count = 0;
    loop {
        if count == 9 {
            break;
        }
        count += 1
    }
    println!("end");
}
```

지금 설명하는 부분은 다른 언어와 다른 부분인데, 다중 loop문이 있을 때
break와 continue를 만나면 현재 위치에 해당하는 반복문을 대상으로 뛰어넘기 되거나
반복문을 탈출하게 되는데, loop의 경우 특정 라벨을 지정하면 해당 라벨에 달린 loop를 대상으로
continue나 break를 지정할 수 있다.

```rust
fn main() {
    let mut count = 0;
    'counting_up: loop {
        println!("count = {count}");
        let mut remaining = 10;

        loop {
            println!("remaining = {remaining}");
            if remaining == 9 {
                break;
            }
            if count == 2 {
                break 'counting_up;
            }
            remaining -= 1;
        }

        count += 1;
    }
    println!("End count = {count}");
}
```


#### b. while
while을 통해서 조건을 통해 반복문을 구현 할 수 있다
```rust
fn main() {
    let mut counter = 0;

    while counter != 10 {
        println!("{number}");

        number += 1;
    }

    println!("end");
}
```

if 문과 동일하게 while 이후에 조건문이 들어오며 조건문이 true일때만 중괄호 안에 코드가 실행된다.

#### c. for
for문을 통해 배열에 대해 값을 하나씩 갖고와서 반복문을 실행시킬 수 있다.
```rust
fn main() {
    let array = [10, 20, 30, 40, 50];

    for element in array {
        println!("{element}");
    }
}
```

위 코드는 배열의 원소들을 하나씩 갖고 와서 처리하며 모두 갖고와서 처리하면 반복문이 종료된다.

# 참고문헌
- [The Rust Programming Language - 주석](https://doc.rust-kr.org/ch03-04-comments.html)
- [The Rust Programming Language - 흐름제어](https://doc.rust-kr.org/ch03-05-control-flow.html)
