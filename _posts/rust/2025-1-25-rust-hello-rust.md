---
title: Rust 입문하기
author: blakewoo
date: 2025-1-25 22:00:00 +0900
categories: [Rust]
tags: [Rust] 
render_with_liquid: false
---

# Rust

## 1. 도입문
요새 Rust가 뜨고 있다.   
누군가 말하길 c나 java같은 언어를 할줄 알면 그렇게 어렵지 않다고는 하는데 또 누군가는 진입장벽이 꽤 있다고 한다.    
최근에 kernel 코드에도 rust가 도입되는 추세라고 하니 이 기회에 언어를 알아두면 좋을 것 같아서 공부하려고 한다.

> ※ 본 포스팅은 C나 JAVA, C++, Javascript 언어를 접해봤거나 아는 사람을 타겟으로 작성되었다.
{: .prompt-tip }

linux를 개발환경으로 써도 좋으나 일단은 여기서는 Window 기준으로 설치하고 설명하겠다.

## 2. 개발 환경 구성
### 1) Microsoft C++ Build Tools 설치
Visual studio가 설치되어있다면 굳이 설치하지 않아도 되고, 별도의 c++ 프로그램을 사용한 적이 있다면
아마 이미 설치되어있을 가능성이 있다. 하지만 없을 경우를 대비해서 
[빌드툴 설치 홈페이지](https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/) 에서
파일을 받아 미리 설치해두는게 좋다.

설치 파일을 받아서 실행할 경우 아래와 같은 화면이 뜬다.

![img_1.png](/assets/blog/rust/install/img_1.png)

여기서 c++을 사용한 테스크톱 개발에만 체크해두면 당장 Rust를 써보는데 문제가 없다.   
해당 패키지를 설치한다.

### 2) 러스트 컴파일러 설치
[러스트 공식 홈페이지](https://www.rust-lang.org/tools/install) 로 가서 Rust 빌드 툴을 받는다.   
당장 보이는 것은 32bit인지 64bit 고르는 버튼이 보일 것이다. 각자 자기 컴퓨터 아키텍처에 맞는 버전으로 누른다.

이후 아래의 창이 뜬다

![img.png](/assets/blog/rust/install/img.png)

여기서 그냥 1번을 입력하면 자동으로 설치된다.

### 2) IDE 설치
여기서는 visual studio code를 설치할것이다. [공식홈페이지](https://code.visualstudio.com/) 에서
visual code를 받아서 설치한다.

### 3) 편의성 도구 설치
그 뒤로 visual studio code가 설치되었다고 전부가 아니다. 이 visual studio code를 사용하는 가장 큰 이유가 추가 플러그인이
강력해서인데, Rust에 대한 문법 교정이나 여타 많은 설정들을 사용하기위해서는 추가 플러그인이 필수이다.

아래는 rust를 사용할때 설치해두면 좋은 플러그인 목록이다.

#### a. rust-analyzer
러스트를 Visual Studio Code에서 사용하기 위해 지원하는 플러그인이라고 볼 수 있다.
러스트 코드 자동 완성과 함수 정의로 이동등 코드 작성을 위한 편의 기능을 제공한다.

#### b. Even Better TOML
rust에서 cargo를 사용하는데 이 cargo가 사용하는 파일이 toml 파일이다.   
이 toml 파일을 보기 쉽게 하이라이트 처리하고 포맷팅해주는 플러그인이다.

#### c. Dependi
패키지 버전관리 플러그인으로 종속성 관리 플러그인이다. 

#### d. CodeLLDB
디버깅을 지원하는 플러그인이다. visual studio enterprise를 사용할 때 쓸 수 있는 디버거와 같이
브레이크 포인트와 메모리 view 기능등을 지원한다.

## 3. Hello Rust!
설치를 완료했으니 Rust를 구동해봐야한다.   
cmd 창을 띄우고 폴더를 만들고자하는 위치로 가서 원하는 폴더이름을 아래와 같이 입력한다.
```
cargo new {폴더 이름}
```
그러면 지정한 이름의 폴더와 함께 rust 프로젝트가 만들어진다.
그리고 아래와 같은 파일과 폴더들이 생성된다.
```
.git
src
.gitignore
Cargo.toml
```
.git과 .gitignore는 git을 사용하기 위한 파일이고
Cargo.toml은 패키지 이름과 버전, 그리고 종속성이 담겨있는 toml 파일이다.
src는 실질적인 rust 코드가 들어가있는 폴더인데, 안에 main.rs 파일이 있다.
main.rs는 아래와 같은 코드로 이루어져있다.
```
fn main() {
    println!("Hello, world!");
}
```
cmd로 해당 프로젝트로 들어가 아래와 같은 작업을 할 수 있다.
### 1) 빌드
```
cargo build
```

프로젝트 폴더 안에 target이라는 폴더가 생기며 안에 debug 폴더안에 exe 파일이 생긴다.

### 2) 실행
```
cargo run
```
프로젝트 폴더 안에 target이라는 폴더가 생기며 안에 debug 폴더안에 exe 파일이 생기는 것 까지 같지만
현재 실행중인 cmd 창에 결과를 띄운다.

### 3) 산출물 삭제
```
cargo clean
```
산출물을 지우는 명령어로 실행시 target 폴더를 지운다.
