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

## 1. 변수와 가변성
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

## 2. 데이터 타입
rust에서도 아래와 같이 타입을 명시하여 변수를 만들 수 있다.
```rust
fn main() {
  let b: f32 = 3.0; // f32형태의 변수
}
```

타입을 명시하지 않는다면 Number literals나 값을 보고 유추하여 타입이 정해진다.

rust의 데이터 타입은 크게 두 종류가 있다.

### 1) 스칼라 타입
하나의 값으로 표현되는 타입을 스칼라 타입이라고 한다.
총 4가지가 있다.

#### a. 정수형
소수점이 없는 숫자로, 부호가 필요없을 때 값의 크기를 2배로 크게 쓸수있는 unsigned형
부호를 사용하는 signed 형이 있다.

<table><thead><tr><th>Length</th><th>Signed</th><th>Unsigned</th></tr></thead><tbody>
<tr><td>8-bit</td><td>i8</td><td>u8</td></tr>
<tr><td>16-bit</td><td>i16</td><td>u16</td></tr>
<tr><td>32-bit</td><td>i32</td><td>u32</td></tr>
<tr><td>64-bit</td><td>i64</td><td>u64</td></tr>
<tr><td>arch</td><td>isize</td><td>usize</td></tr>
</tbody></table>

변수에 값을 대입할때 아래와 같이 접두사나 접미사를 통해서 입력하면 해당 type으로 인지시킬수 있다.

<table><thead><tr><th>Number literals</th><th>Example</th></tr></thead><tbody>
<tr><td>Decimal</td><td><code>98_222</code></td></tr>
<tr><td>Hex</td><td><code>0xff</code></td></tr>
<tr><td>Octal</td><td><code>0o77</code></td></tr>
<tr><td>Binary</td><td><code>0b1111_0000</code></td></tr>
<tr><td>Byte (<code>u8</code> only)</td><td><code>b'A'</code></td></tr>
</tbody></table>

#### b. 부동소수점
총 두 가지 타입이 있다.
f32와 f64가 그 두가지인데, 각각 32bit와 64bit의 크기를 가지며
기본 타입은 f64이다. 이는 둘다 대략 비슷한 성능에 f64가 좀 더 정밀하게 표현이 가능하기 때문이다.
아래는 그 표기의 예시이다.

```rust
fn main() {
  let a = 2.0; // f64
  let b: f32 = 3.0; // f32
}
```

#### c. boolean
다른 언어들처럼 true 혹은 false를 나타낼 수 있으며 아래와 같이 지정할 수 있다.
```rust
fn main() {
  let a: bool = true;
}
```

#### d. 문자 타입
rust의 문자 타입인 ```char```는 Unicode Scalar를 표현하는 값이다.
String 타입과는 다른 타입이기 때문에 주의해야하며
작은 따옴표로 아래와 같이 표기 할 수 있다.
```rust
fn main() {
   let c = 'z';
   let z = 'ℤ';
}
```

#### e 문자열 타입
한 개의 문자로 이루어진 문자 타입과는 다르게 문자열 타입은 여러 문자로 이루어져있다.
또한 한번 지정하면 변경할 수 없다.
문자열 타입은 아래와 같이 쓴다.
```rust
fn main() {
    let s = "hello";
    println!("{}", s);
}
```

아래와 같이 사용하기도 한다.
```rust
fn main() {
    let s: &'static str = "hello";
    println!("{}", s);
}
```


### 2) 복합 타입

#### a. 튜플
값들을 집합시켜서 만들 수 있는 타입이다. 일반적으로 몇 개의 숫자를 집합시켜 튜플로 만드는데
튜플에 포함되는 각 값의 타입이 동일할 필요는 없다.
아래는 그 예시이다.
```rust
fn main() {
    let tup: (i32, f64, u8) = (500, 6.4, 1);
}
```
이렇게 만들어진 튜플을 해제하기 위해서는 패턴 매칭을 통해 튜플의 값을 해체하여 사용하면 된다.

```rust
fn main() {
    let tup = (500, 6.4, 1);

    let (x, y, z) = tup;

    println!("The value of y is: {}", y);
}
```

혹은 원하는 값의 색인을 넣는 것으로 튜플 요소에 직접 접근할 수 있따.
예시는 아래와 같다.

```rust
fn main() {
    let x: (i32, f64, u8) = (500, 6.4, 1);

    let five_hundred = x.0;

    let six_point_four = x.1;

    let one = x.2;
}
```

#### b. 배열
여러 값들의 집합체를 배열이라고 한다. 튜플이랑 다른 점은 배열의 모든 요소는 같은 타입이어야하며
배열은 고정된 길이를 갖는다.

선언하는 방법은 아래와 같다.
```rust
fn main() {
    let a = [1, 2, 3, 4, 5];
}
```

선언된 배열은 다른 언어에서 배열에 접근하듯이 index 값으로 접근이 가능하다.
```rust
fn main() {
    let a = [1, 2, 3, 4, 5];

    let first = a[0];
    let second = a[1];
}
```

index 값은 0부터 시작하며 만약 유효하지 않은 값을 넣는다면 실행중에 에러가 발생하게 된다.

### 3) String 타입
문자열 타입에 대해서 언급했는데 왜 String 타입이 나오는지 의문스러울 것이다.   
이는 문자열 타입의 경우 데이터를 바꿀수없고, 별도의 함수를 통해서 가공할 수 있는 반면
이 String 타입은 값을 바꿀수 있으며 힙 영역에 저장된다는 점이 다르기 때문에 별도의 타입으로 분류해두었다.   
세부적인 내부 구조의 경우 소유권에 대해 포스팅할 때 추가적으로 포스팅할 예정이며
String 타입에 대해 정의는 아래와 같이 한다.

```rust
fn main() {
    let mut s = String::from("hello");
    println!("{}", s);
}
```


# 참고문헌
- [The Rust Programming Language - Variables and Mutability](https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html)
