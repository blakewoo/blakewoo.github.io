---
title: Rust - 소유권
author: blakewoo
date: 2025-1-31 15:30:00 +0900
categories: [Rust]
tags: [Rust] 
render_with_liquid: false
---

# Rust 문법

## 1. 소유권
### 1) 개요
몇몇 언어는 Garbage Collector를 운용해서 쓰지 않는 메모리를 주기적으로 체크해서 회수하게 하고
몇몇 언어는 명시적으로 메모리를 할당하고 풀어줘야한다. 위 두 가지 방식 모두 장단점이 있다.

Garbage Collector는 별도로 메모리 관리에 크게 신경을 써도 되지 않아도 되는 반면
실행 간의 별도의 스레드가 돌아가서 성능에 영향을 준다.

명시적으로 메모리를 할당하고 풀어주는 방식은 구현자가 어떻게 구현하느냐에 따라
메모리를 효율적으로 사용할 수 있는 장점이 있지만 반대로 구현자의 실수에 취약하여 메모리 누수가 일어날 수 있다는 점이다.

위의 방식의 경우 이러나저러나 문제가 생기므로 rust에서는 소유권 개념의 새로운 메모리 관리 방식을 도입했다.

### 2) 규칙
큰 규칙은 아래의 3개다.

- 러스트에서, 각각의 값은 소유자 (owner) 가 정해져 있다.
- 한 값의 소유자는 동시에 여럿 존재할 수 없다.
- 소유자가 스코프(Scope) 밖으로 벗어날 때, 값은 버려진다 (dropped).

언급하기 전에 일단 Scope의 개념에 대해서 미리 알아야한다. 아래의 예시를 보자.
```rust
fn main() {
  { // ---- (1)   
    let scope_val = 1; // ---- (2)
  } // ---- (3) 
}
```

(1) 에서 scope_val은 선언되지 않아 유효하지 않다.   
(2) 에서 scope_val은 선언되어 유효하다.   
(3) 에서 scope_val은 스코프가 종료되어 더 이상 유효하지 않다.   

사실 다른 언어들에서 말하는 Scope와 다를바없다. 중괄호를 기준으로 scope가 잡힌다고 생각하면 편하다.   

또한 다른 언어에서 값을 복사하는 방식으로 깊은 복사와 얕은 복사를 말한다.
하지만 rust에서는 copy와 move로 정의할 수 있다.
copy는 매개변수나 값을 넘길때 값을 복사해서 넘기는 것이고,
move는 해당 값이 있는 주소의 소유권을 이전하는 것이라고 볼 수 있다.
이 부분에 대해서는 아래에서 좀 더 세부적으로 설명하겠다.

#### a. 러스트에서, 각각의 값은 소유자 (owner) 가 정해져 있다.
```rust
fn main() {
    let s1 = String::from("hello");
}
```
위의 코드를 보자 s1이 가진 값은 String 데이터 타입이다. 문자열과는 다른 방식으로 작동하는데
문자열의 경우 stack에 저장되어있는 반면, 이 String 데이터 타입은 힙 영역에 저장되어서 스택에 포인터로 걸려있다.

![img.png](/assets/blog/rust/ownership/img.png)

위의 그림을 볼때 s1은 ```String::from("hello");``` 이 값을 소유하고 있고, 즉 소유자라고 볼 수 있다.
(앞에 ```let s1```을 선언하지 않아도 경고만 뜨지 build가 되긴한다, 하지만 안 쓸거면 정의할 의미가 없다)

#### b. 한 값의 소유자는 동시에 여럿 존재할 수 없다.
아래의 코드를 보자
```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;
}
```
s2가 s1 값을 받아왔다. 즉 s2에 s1이 갖고 있던 주소가 들어갔다.
이 경우 다른 언어들의 경우를 생각할때 아래와 같은 형태가 되었다고 생각할 수 있다.

![img_1.png](/assets/blog/rust/ownership/img_1.png)

위와 같은 형태라면 얕은 복사라고 부를 수 있겠지만 rust는 소유권이 이동(move)한다고 했다.
따라서 위 그림과 같은 형태가 아니라 아래와 같은 형태가 된다.

![img_2.png](/assets/blog/rust/ownership/img_2.png)

때문에 s1에 대해서 사용하려들면 에러가 발생한다.

이 경우 s1이 갖고 있던 데이터의 소유권이 s2에게 넘어간 것으로 생각할 수 있으며
move 라고 부른다.

#### ※ 이동이 안되는 경우
몇몇 데이터 타입의 경우 이동이 아닌 복사가 되는데 그 타입들은 아래와 같다.

- 모든 정수형 타입 (예: u32)
- true, false 값을 갖는 논리 자료형 bool
- 모든 부동 소수점 타입 (예: f64)
- 문자 타입 char
- Copy 가능한 타입만으로 구성된 튜플 (예를 들어, (i32, i32)는 Copy 가능하지만 (i32, String)은 불가능)

#### ※ 값 복사하는 법
물론 값 자체를 복사해서 별도의 인스턴스를 가지고 있을 수 있다.   
이는 clone 메소드를 사용하며 된다.

```rust
    let s1 = String::from("hello");
    let s2 = s1.clone();

    println!("s1 = {}, s2 = {}", s1, s2);
```

이러면 별도의 힙 메모리가 할당되고 동일한 내용이 복사되어 s2에 연결된다.


#### c. 소유자가 스코프(Scope) 밖으로 벗어날 때, 값은 버려진다 (dropped).
아래의 코드를 보자
```rust
fn main() {
  {
    let s1 = String::from("hello");
  }
  println!("{}", s1);
}
```

중괄호를 기준으로 s1이 scope를 벗어났기 때문에 값이 버려져서 원래 생성되어있던 ```String::from("hello");```값은
사용할 수 없다.

## 2. 참조와 대여
위에서 스택에 있는 값이 아닌 힙에 있는 데이터를 사용할 때 이동이 일어난다고 했다.
이 경우는 함수에 인자로 넘길 때도 동일한 일이 생긴다. 아래의 경우가 그러한 예시이다.

```rust
fn main() {
    let s1 = String::from("hello"); // String 타입의 hello가 생성

    let s2 = something_doing(s1);  // hello 데이터의 소유권인 somting_doing으로 넘어감
    
    // 해당 위치에서는 s1을 사용 할 수 없음, 소유권 이전되어버림
} 

fn something_doing(a_string: String) -> String { 
    a_string  // 입력 받은 a_string을 반환
}
```

하지만 함수에 인자로 넘겨준 이후에도 해당 값을 계속 사용해야할때가 있다.
이때 필요한 것이 바로 참조자(reference)라는 것이다.
참조자 (reference) 는 해당 주소에 저장된 데이터에 접근할 수 있도록 해주는 주솟값에 해당하는, 포인터와 같은 것이다.
아래의 예시를 살펴보자.

```rust
fn main() {
    let s1 = String::from("hello");

    let len = calculate_length(&s1);

    println!("The length of '{}' is {}.", s1, len);
}

fn calculate_length(s: &String) -> usize {
    s.len()
}
```

위의 코드와 달리 인자 부분에 & 문자가 들어간 것을 알 수 있다. 또한 함수 선언부에도 &가 있는 것을 확인할 수 있다.
이렇게 선언해두면 s1에 대해 참조자가 넘어가서 이후에도 s1을 그대로 사용할 수 있다.
이렇게 참조자를 만드는 것을 대여라고 한다.

일반적으로 참조자를 이용해서 값을 넘긴 것은 수정할 수 없다.   
따라서 아래의 코드는 컴파일시 에러를 반환한다.

```rust
fn main() {
    let s = String::from("hello");

    change(&s);
}

fn change(some_string: &String) {
    some_string.push_str(", world");
}
```

만약에 해당 값을 변경하고 싶다면 참조자의 값을 변경가능한 형태로 선언해야한다.
이를 가변 참조자라고 하며 선언방법은 아래와 같다.

```rust
fn main() {
    let mut s = String::from("hello");

    change(&s);
}

fn change(some_string: &String) {
    some_string.push_str(", world");
}
```

이러한 가변 참조자라고 만능은 아니다.   
이 가변참조자를 미리 하나 만들었다면 그 값에 대한 참조자는 더 이상 만들수 없다.   
따라서 아래 코드는 가변 참조자가 있음에도 추가 참조자를 만들었다고 에러가 발생한다.

```rust
 let mut s = String::from("hello");

  let r1 = &mut s;
  let r2 = &mut s;

  println!("{}, {}", r1, r2);
```

## 3. 댕글링 포인터(Dangling Pointer)
이미 해제된 메모리를 가리키고 있는 걸 댕글링 포인터라고 한다. 포인터를 사용하는 언어에서는 자주 일어날 수 있는 일인데
rust에서는 이러한 댕글링 참조의 경우를 미리 캐치하여 에러를 발생시켜 댕글링 참조가 발생하지 않도록 보장해준다.
아래와 같은 경우는 댕글링 참조가 발생했다고 에러가 발생한다.

```rust
fn main() {
    let reference_to_nothing = dangle();
}

fn dangle() -> &String {
    let s = String::from("hello");

    &s
}
```

dangle 함수가 끝나는 순간 s는 없어지므로, s에 대한 참조자는 이미 해제된 영역을 가르키게 된다.
이런 경우 아예 s에 대한 소유권을 넘겨주어야하기 때문에 &s가 아닌 s를 반환하면 된다.

## 4. 슬라이스(Slice)
슬라이스는 컬렉션을 통째로 참조하는 것이 아닌 컬렉션의 연속된 일련의 요소를 참조하도록 해주는 참조자이다.
문자열의 경우 아래와 같이 사용한다.

```rust
fn main() {
  let s = String::from("hello");
  
  let len = s.len();
  
  let slice = &s[0..2]; // 0부터 2까지, 즉 "hel"
  let slice = &s[..2];
  
  let slice = &s[3..len]; // 3부터 끝까지, 즉 "lo"
  let slice = &s[3..];
  
  let slice = &s[0..len]; // 0부터 끝까지, 즉 "hello"
  let slice = &s[..];
}
```

ASCII 문자의 경우 이렇게 사용할 수 있지만 UTF-8이나 다른 포맷을 사용할시에 적절한 경계를 사용해야 프로그램에
문제가 없어진다.

i32 형태의 배열에도 똑같이 작동한다.

```rust
fn main() {
  let a = [1, 2, 3, 4, 5];
  
  let slice = &a[1..3];
}
```

이렇게 생성된 슬라이스는 함수의 매개변수로도 사용할 수 있으며 불변형 참조자와 동일하게 사용할 수 있다.


# 참고문헌
- [The Rust Programming Language - 소유권](https://doc.rust-kr.org/ch04-01-what-is-ownership.html)
- [The Rust Programming Language - 참조, 대여](https://doc.rust-kr.org/ch04-02-references-and-borrowing.html)
- [The Rust Programming Language - 슬라이스](https://doc.rust-kr.org/ch04-03-slices.html)
