---
title: Rust - 변수
author: blakewoo
date: 2025-1-26 23:00:00 +0900
categories: [Rust]
tags: [Rust] 
render_with_liquid: false
---

# Rust 문법

> ※ 본 포스팅은 추가적으로 업데이트 될 예정이다.
{: .prompt-tip }

## 변수와 가변성
Rust에서 변수는 기본적으로 불변형이다.   
그렇다고 상수와는 또 다르다. 실질적인 예시를 아래와 같다.

```rust
fn main() {
    let x = 5;
    println!("The value of x is: {}", x);
    x = 6;
    println!("The value of x is: {}", x);
}
```

위 코드를 실행시키면 에러가 난다!   
이는 위에서 말한 불변형과 관련이 있다.

위의 코드에서 x 변수에 다시 값을 주고 싶으면 mut를 붙여야한다.

```rust
fn main() {
    let mut x = 5;
    println!("The value of x is: {}", x);
    x = 6;
    println!("The value of x is: {}", x);
}
```

x 변수는 새로 값이 할당될 수 있다는 것을 알려주는 표시인 것이다.
이 mut 값이 없다면 x에 대한 새로운 값을 할당할 수 없다.
하지만 아래와 같은 경우는 가능하다.

```rust
fn main() {
    let x = 5;
    let x = x + 1;
    let x = x * 2;
    println!("The value of x is: {}", x);
}
```

이 부분은 실제 상수인 const와는 다른 부분인데, 원래 있던 변수를 사용한
값이라면 위와같이 재 할당이 가능하다.
이를 shadowing이라고 한다.

다만 아래와 같은 경우는 안된다.

```rust
fn main() {
    let mut x = 5;
    x = x.len();
    println!("The value of x is: {}", x);
}
```

mut 지시어가 있다면 같은 이름을 사용하는 변수의 유형을 바꿀 수 없기 때문이다.



# 참고문헌
- [The Rust Programming Language - Variables and Mutability](https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html)
