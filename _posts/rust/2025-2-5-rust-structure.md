---
title: Rust - 구조체
author: blakewoo
date: 2025-2-5 17:50:00 +0900
categories: [Rust]
tags: [Rust] 
render_with_liquid: false
---

# Rust 문법

## 1. 구조체

### 1) 개요 
c언어에서도 그러하듯 rust에서도 구조체를 지원한다.   
특정 의미 있는 값들을 묶고 정의하는데 유용한데, Rust에서는 어떻게 구조체를 사용할 수 있는지 알아볼 것이다.

### 2) 구조체 정의
구조체 자체를 쓰는 법은 어렵지 않다.   
struct라고 앞에 선언하고 뒤에 이름을 붙인 뒤 중괄호를 치고 각각 속성에 대한 이름과
타입을 명시하는게 기본 형태이다.

아래는 User라는 구조체를 임의로 정의해본 예시이다.
```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}
```

> ※ 여기서 String 말고 str을 쓸순 없냐고 물을수 있는데 이는 이후에 lifetime까지 알아본 이후에나 사용 가능할 것이다.
{: .prompt-tip }

위와 같이 정의 했으면 아래와 같이 사용해 볼 수 있다.

```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

fn main() {
    let user1 = User {
        active: true,
        username: String::from("someusername123"),
        email: String::from("someone@example.com"),
        sign_in_count: 1,
    };
}
```

각각의 인자값을 부여했다면 문제없이 인스턴스가 만들어지며 점을 통해서 각각의 속성에
개별적으로 엑세스할 수도 있다.
```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

fn main() {
    let user1 = User {
        active: true,
        username: String::from("someusername123"),
        email: String::from("someone@example.com"),
        sign_in_count: 1,
    };
    
    user1.email = String::from("someone2@example.com")
}
```

특정 함수에 대해 동일한 이름이 있다면 아래와 같이 축약해서 인스턴스를 생성할 수도 있다.
```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

fn build_user(email: String, username: String) -> User {
    User {
        active: true,
        username, // username 의 이름이 같아 축약됨
        email, // email 의 이름이 같아 축약됨
        sign_in_count: 1,
    }
}

fn main() {
    let user1 = build_user(
        String::from("someone@example.com"),
        String::from("someusername123"),
    );
}
```

만약 동일한 구조체이나 특정 부분만 다른 값으로 새로운 인스턴스를 만들고 싶을 경우
아래와 같이 축약하여 선언 가능하다.

```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

fn main() {
    let user1 = User {
        email: String::from("someone@example.com"),
        username: String::from("someusername123"),
        active: true,
        sign_in_count: 1,
    };

    let user2 = User {
        email: String::from("another@example.com"),
        ..user1 // email을 제외한 값까지 모두 동일하여 축약하여 선언
    };
}
```

> ※ 여기서 user1의 email, username은 String이므로 소유권이 user2에 넘어가버려 사용이 불가하다. 단 나머지는 속성은 사용 가능하다.
{: .prompt-tip }

### 3) 튜플 구조체 정의
굳이 속성에 이름을 만들지 않아도 될 경우
가령 x,y,z 좌표라던지, RGB 같은 것들을 구조체로 만들고 싶을 경우 아래와 같이 선언가능하다.

```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

fn main() {
    let black = Color(0, 0, 0);
    let origin = Point(0, 0, 0);
}
```

속성들이 동일한 데이터 타입을 가지지만 다른 구조체이기에 서로 대입이 불가하며 튜플로써의
기능은 그대로 갖고있기 때문에 해체할 수도 있고 인덱스로 개별값에 접근 역시 가능하다.

```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

fn main() {
    let color_number = Color(22, 0, 0);
    let origin_point = Point(0, 0, 0);
    
    println!("{}",color_number.0);
    println!("{}",origin_point.0);
    
    //각각 22와 0이 출력된다.
}
```

### 4) 필드 없는 유사 유닛 구조체
필드가 아예 없는 구조체를 정의할 수도 있는데, 이를 유사 유닛 구조체(unit-like structs)라 지칭한다.   
이 부분은 추가적으로 더 알아볼 예정이고, 정의법만 알아보자

```rust
struct AlwaysEqual;

fn main() {
    let subject = AlwaysEqual;
}
```

## 2. 메서드
구조체안에 함수를 넣을 수도 있다. 이런 함수는 연관 함수(associated function)라고 부른다.
연관 함수는 구조체 자기자신을 인자로 가질수 있는 이런 함수는 메서드(Method)라고 부른다.

아래의 예시를 보자
```rust
#[derive(Debug)]
struct Location {
    x: u32,
    y: u32
}

impl Location{
    fn distance(&self) -> f64 {
        let sum = self.x as f64 + self.y as f64;
        sum.sqrt()
    }
}

fn main() {
    let point = Location {
        x:1,
        y:2
    };

    println!("{}",point.distance())
}
```

Location이라는 구조체를 선언하였고 impl 식별자를 이용하여 Location 구조체에
거리를 계산하는 distance 함수를 추가하였다.
&self 값을 인자로 줌으로써 현재 있는 구조체의 값을 ```self.``` 형태로 불러와서 사용할 수 있다. 

이렇게 impl를 이용하여 구조체에 함수를 추가 할 수 있으며 꼭 한 개의 impl만 있을
필요는 없다.

```rust
#[derive(Debug)]
struct Location {
    x: u32,
    y: u32
}

impl Location{
    fn distance(&self) -> f64 {
        let sum = self.x as f64 + self.y as f64;
        sum.sqrt()
    }
}

impl Location{
    fn read_x(&self) -> u32 {
        self.x
    }
        
    fn read_y(&self) -> u32 {
        self.y
    }
}

fn main() {
    let point = Location {
        x:1,
        y:2
    };

    println!("{}",point.distance())
}
```

모든 연관 함수가 method 일 필요는 없다. 즉, 구조체 자기자신을 인자로 가질 필요는 없다.    
이런 경우는 구조체의 새 인스턴스를 반환하는 생성자로 자주 활용된다.    
생성자로 사용하는 예시는 아래와 같다.

```rust
#[derive(Debug)]
struct Location {
  x: u32,
  y: u32
}

impl Location{
  fn x_is_y(position: u32) -> Self  {
    Self  {
        x:position,
        y:position
    }
  }
}
fn main() {
  let xy = Location::x_is_y(1);
}
```


# 참고문헌
- [The Rust Programming Language - 구조체 정의 및 인스턴스화](https://doc.rust-kr.org/ch05-01-defining-structs.html) 
