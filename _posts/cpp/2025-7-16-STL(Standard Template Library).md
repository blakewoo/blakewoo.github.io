---
title: C++ - 자주 쓰는 STL
author: blakewoo
date: 2025-7-16 20:00:00 +0900
categories: [C++]
tags: [C++, STL] 
render_with_liquid: false
use_math: true
---

# C++ 자주 쓰는 STL
## 1. STL 개요
STL(Standard Template Library)는 프로그래밍 언어를 위해 처음 설계한 소프트웨어 라이브러리이다.   
원래라면 C++에만 있는건 아니라는 뜻이다. 하지만 C++ 표준 라이브러리에 많은 영향을 끼쳤으며 이는 
컨테이너, 알고리즘, 반복자, 기능으로 구분된다.

### 1) 컨테이너(Container)
이 컨테이너는 순서 컨테이너와 연관 컨테이너가 있다. 

#### a. 표준 순서 컨테이너
말 그대로 순서가 있는 것들이다.
vector, deque, list

#### b. 표준 연관 컨테이너
순서는 없지만 연관되어있는 것들이다.
set, map, hash_set, hash_map

### 2) 반복자(Iterators)
반복자는 어떤 값을 반복적으로 처리하기 위한 것으로 영어로는 Iterators를 말한다.   
이 반복자는 아래와 같이 5 종류가 있다.

- 입력 반복자 (값 시퀀스를 읽는 데만 사용 가능)
- 출력 반복자 (값 시퀀스를 쓰는 데만 사용 가능)
- 순방향 반복자 (읽고, 쓰고, 앞으로 이동할 수 있음)
- 양방향 반복자 (순방향 반복자와 유사하지만 뒤로 이동할 수도 있음)
- 임의 접근 반복자 (한 작업에서 원하는 수의 단계를 자유롭게 이동할 수 있음).

### 3) 알고리즘(Algorithm)
검색이나 정렬과 같은 작업을 수행 할 수 있는 알고리즘을 말한다.
- sort, binary search

### 4) 함수 객체(Function Object)
STL에는 함수 호출 연산자를 오버로드하는 클래스가 포함되어있는데 이러한 클래스의 인스턴스를 함수 객체라고 한다.
이 함수 객체는 연관된 함수의 동작을 매개변수화할 수 있도록 하며, 함수와 함께 연관된 함수 객체 정보를 유지하는데 사용될 수 있다.

그냥 들으면 이게 뭔 소린가 싶은데 그냥 해당 STL을 함수 처럼 쓸수 있게 해준다는 뜻이다.   
아래의 예시를 살펴보자.

```cpp
#include <iostream>

class Add {
public:  // 중요! class는 기본 접근이 private이므로 명시적으로 public으로 열어줘야 함
    int operator()(int a, int b) const {
        return a + b;
    }
};

int main() {
    Add add;
    std::cout << add(3, 5) << std::endl;  // 출력: 8
}
```

위와 같이 class를 선언한다면 객체이지만 함수같이 사용할 수 있으며 이를 함수 객체라고 한다.

## 2. 자주 쓰는 STL
### 1) Array
#### a. 선언
사용하기 위해서는 array 헤더를 추가해야한다.
```cpp
#include<array>

array<int, 5> arr; // int형으로 크기 5의 array를 선언
```

#### b. 값 추가
std::array는 고정 크기이므로 원소를 추가할 수 없지만, 인덱스를 통해 값을 할당한다.
```cpp
arr[0] = 10;
arr[1] = 20;
arr.at(2) = 30; // 범위 체크가 포함된 접근
```

#### c. 업데이트
인덱스를 통해 직접 값을 변경한다.
```cpp
if (!arr.empty()) {
    arr[3] = 40;      // operator[] 사용
    arr.at(4) = 50;   // at() 사용
}
```

> ※ 성능을 살펴볼때 array보단 vector가 더 낫다는 말도 있다.
{: .prompt-tip }


#### d. 삭제
std::array는 고정 크기이므로 원소를 삭제할 수 없다. 대신 모든 원소를 초기화하거나 기본값으로 채운다
```cpp
arr.fill(0); // 모든 원소를 0으로 초기화
```

### 2) Vector
#### a. 선언
사용하기 위해서는 vector 헤더를 추가해야한다.
```cpp
#include<vector>

vector<int> vec; // int형 가변크기 vector 선언
vector<int> vec2 = {1,2,3}; // int형 가변크기 vector에 1,2,3으로 초기화
vector<int> vec3(vec2); // vec2와 동일한 값으로 초기화
```

Vector는 가변 크기지만 미리 크기를 할당해둘수도 있다.
```cpp
vector<int> vec1(4) // 크기 4로 생성
vector<int> vec2(10,4); // 크기 10에 4로 초기화
vector<vector<int>> vec3(9,vector<int>(8,6) // 크기 8이고 6으로 초기화된 벡터들이 총 9개 (8x9, 2차원 벡터) 
vector<vector<int>> vec4 = {
  {1,2,3},
  {4,5,6},
  {7,8,9}
} // 초기화 리스트를 이용하여 2차원 벡터 초기화
```

#### b. 값 추가
값을 뒤에서 넣기와 지정해서 해당 위치에 넣기가 있으며, 지정해서 해당 위치에 넣기보다 뒤에서 넣기가 성능이 더 좋다.

```cpp
vec.push_back(10);               // 맨 뒤에 값 추가
vec.insert(vec.begin(), 5);      // 지정한 위치에 값 추가
```

#### c. 업데이트

```cpp
if (!vec.empty()) {
    vec[0] = 15;         // operator[] 사용
    vec.at(1) = 20;      // at() 사용
}
```

#### d. 삭제

```cpp
// 특정 위치의 원소 삭제
if (!vec.empty()) vec.erase(vec.begin() + 1);

// 제일 뒤의 원소 삭제
vec.pop_back();

// 값이 10인 모든 원소 삭제 (erase-remove idiom)
vec.erase(std::remove(vec.begin(), vec.end(), 10), vec.end());

// 전체 삭제
vec.clear();
```


### 3) Set
#### a. 선언
사용하기 위해서는 set 헤더를 추가해야한다.
```cpp
#include<set>

set<int> intSet; // int형으로 set 선언
set<int> intSet2 = {1,2,3}; //  set을 1,2,3으로 초기화
set<int> intSet3(intSet2); // set을 intSet2로 생성
```

#### b. 값 추가

```cpp
intSet.insert(10);
intSet.insert(20);
```

#### c. 값 탐색
```cpp
set<int> datas = {1,2,3,4,5};

auto targetIndex = datas.find(3);

if(targetIndx != datas.end()){
 // 해당 값이 set에 있음
}
else{
 // 해당 값이 set에 없음
}
```

※ 사실 set을 쓸 때 뭔가를 넣어두고 안에 값이 있는지 없는지 체크하려고 쓰는건데,
위와 같이 쓰면 매우 번거롭다. 그래서 나는 개인적으로 별도의 has 함수를 만들어서 사용한다.

```
#include <set>

template <typename T>
bool has(const std::set<T>& s, const T& value) {
    return s.find(value) != s.end();
}
```

아니면 별도의 wrapper class를 만들어서 사용할 수도 있다.

```cpp
#include <iostream>
#include <set>

template <typename T>
class SetWithHas {
private:
    std::set<T> internalSet;

public:
    // 기본 생성자
    SetWithHas() = default;

    // insert 위임
    void insert(const T& value) {
        internalSet.insert(value);
    }

    // erase 위임
    void erase(const T& value) {
        internalSet.erase(value);
    }

    // 해당 값이 있는지 체크
    bool has(const T& value) const {
        return internalSet.find(value) != internalSet.end();
    }

    // 반복자 위임
    typename std::set<T>::const_iterator begin() const {
        return internalSet.begin();
    }

    typename std::set<T>::const_iterator end() const {
        return internalSet.end();
    }

    // 갯수 반환
    std::size_t size() const {
        return internalSet.size();
    }
    
    // 필요하다면 추가 함수 가능
};
```

하지만 SET을 상속받거나 SET을 직접 고치는건 권장되지 않는다.

#### d. 업데이트
Set은 키 기반으로 정렬된 컨테이너이므로 직접 수정할 수 없다. 값을 변경하려면 기존 값을 지우고 새 값을 삽입한다.

```cpp
int oldValue = 10;
int newValue = 15;
auto pos = intSet.find(oldValue);
if (pos != intSet.end()) {
    intSet.erase(pos);
    intSet.insert(newValue);
}
```

#### e. 삭제

```cpp
intSet.erase(20);           // 값이 20인 요소 삭제

intSet.clear();             // 모든 요소 삭제
```


### 4) Map
#### a. 선언
사용하기 위해서는 map 헤더를 추가해야한다.
```cpp
#include<map>

map<int, int> intMap; // int형을 키와 값으로 갖는 map을 선언
map<int, int> intMap2 = {{1,3}}; // int형키 1과 int형 value 3을 갖는 map을 선언
```

#### b. 값 추가

```cpp
// insert 사용
intMap.insert(std::make_pair(1, 100));
// 또는 operator[] 사용
intMap[2] = 200;
```


#### c. 업데이트
```cpp
intMap[2] = 250; // operator[]로 key가 2인 값 변경
```

#### d. 삭제

```cpp
intMap.erase(1);    // 키 1을 삭제

intMap.clear(); // 전체 삭제
```


> ※ 추가 업데이트 예정
{: .prompt-tip }

# 참고자료
- [위키독스 - C++ 이야기](https://wikidocs.net/25044)
- [c++20 공식문서](https://isocpp.org/files/papers/N4860.pdf)
- [위키백과 - Standard Template Library](https://en.wikipedia.org/wiki/Standard_Template_Library)
- 코딩테스트 합격자 되기 C++ 편. 자료구조, 알고리즘, 빈출 100문제로 대비하는 코테 풀패키지, 박경록, (주)골든 래빗
