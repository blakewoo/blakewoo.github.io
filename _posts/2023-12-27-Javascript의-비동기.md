---
title: Javascript의 비동기
author: blakewoo
date: 2023-12-27 20:00:00 +0900
categories: [javascript]
tags: [javascript, Web, nodejs]
render_with_liquid: false
---

Javascript는 기본적으로 싱글 스레드에서 구동된다.
정확하게는 Javascript가 구동되는 자바스크립트 엔진이 싱글 스레드에서 구동된다.
원래 자바스크립트가 나왔을 당시에 멀티 프로세서가 드물기도 했고,
자바스크립트 자체가 웹 프론트 엔드에서 사용하기 위해 만들어진 언어이니 만큼 블로킹으로 인한 프론트엔드 성능 저하를 막아야했다.
그렇기 때문에 자바스크립트가 차용한 것은 비동기 실행 방식이다.

이 비동기 실행방식이라는 것에 대해서 이해 하기 위해서는 자바스크립트의 구동 구조에 대해서 알아야한다.
자바스크립트 엔진의 내부는 크게 두 영역으로 나뉜다.
1. 콜 스택   
   소스코드 평가 과정에서 생성된 실행 컨텍스트가 추가되고 제거되는 스택이 콜 스택이다.
   함수 호출시 함수 실행 컨텍스트가 순차적으로 콜 스택에 푸시되어 순차적으로 실행된다. 단 하나의 콜스택만 있기 때문에
   콜스택에서 실행 중인 실행 컨텍스트가 종료되어 콜 스택에서 제거 되기 전까지 다른 태스크가 실행되지 않는다.   

2. 힙   
   객체가 저장되는 메모리 공간이다. 콜 스택에서 실행되는 실행 컨텍스트는 힙에 저장된 객체를 참조한다.
   객체의 크기가 미리 정해져있지 않기 때문에 동적 할당해야하며 때문에 힙은 구조화 되어있지 않다.

이렇게 콜 스택과 힙으로 이루어진 자바스크립트 엔진은 단순히 태스크 요청시 콜 스택을 통해 요청한 작업을
순차적으로 실행만 한다. 그러므로 호출 스케중링이나 콜백 함수의 등록 같은 것들은 브라우저나 Nodejs가 담당한다.

이러한 브라우저나 Nodejs에서는 스케줄링을 위해 태스크 큐와 이벤트 루프를 제공한다.
1. 태스크 큐
   setTimeout이나 setInterval과 같은 비동기 함수의 콜백 함수 또는 이벤트 핸들러가 일시적으로 보관되는 영역이다.
2. 이벤트 루프
   이벤트 루프는 콜 스택에 현재 진행 중인 실행 컨테스트가 있는지 태스크 큐에 대기 중인 함수가 있는지 반복해서 확인하며
   만약 콜스택이 비어있고 태스크 큐에 대기중인 함수가 있다면 해당 태스크를 콜스택으로 이동한다.

사실 비동기 문제는 이러한 이벤트 루프와 태스크 큐의 스케줄링에 의해서 발생한다.
아래의 예시를 살펴보자.

````javascript
function a() {
    console.log("a")
}

function b() {
    console.log("b")
}

a()
b()
````

위의 사례는 "a"가 출력되고 "b"가 출력된다.
하지만 아래의 사례는 다르다.

````javascript
function a() {
    console.log("a")
}

function b() {
    console.log("b")
}

setTimeout(a,100);
b()
````

이 경우 "b"가 출력되고 100ms 이후 "a"가 출력된다.
왜 "a"가 먼저 나오지 않는 것일까? 그건 실행순서와 관련이 있다.
아래 실행 순서를 보자
1. setTimeout이 콜스택에 올라와 실행된다. a를 호출 스케줄링하고 setTimeout은 종료된다. 지정 시간이 지난 뒤
   a 함수를 실행 시키는것은 브라우저 혹은 Nodejs의 일이다.
2. b가 실행된다.
3. 타이머 100ms가 지나고 a가 콜 스택에 올라와 실행된다.

이렇게 비동기 함수는 Nodejs나 브라우저에서 스케줄링하여 다른 것들을 먼저 실행시켜버린다.
이런 비동기 함수는 file에 관한 것이나, 네트워크를 타고 서버에 요청하는것이나 DB에 요청하는 것등 여러가지가 있다.

그러면 비동기로 쓸수 밖에 없는 것일까? 그건 아니다.
비동기를 동기 처리할 수 있는 방법은 3가지가 있다.


1. 콜백 함수   
   나중에 실행 되어야 할 함수를 콜백 함수로 넘겨주면 동기적인 작업이 가능하다.
   ```javascript
        const a = (callback) => {
            setTimeout(() => {
                console.log("a");
                callback();
            }, 3000);
        };
        
        const b = () => { console.log("b"); };
        
        a(b);
   ```
   하지만 이 경우 너무 함수가 많아지면 길어지고 복잡해져서 이른바 콜백 지옥이라고 불리는 현상이 일어난다.


2. Promise   
   원래 동기 처리를 위해 만들어진 객체로, 처리 완료시 resolve, 문제 발생시 reject 함수를 호출하여 동기 처리를
   할수 있다.
   ```javascript
        const a = () => {
            return new Promise((resolve, reject) => {
                setTimeout(() => {
                    console.log("a");
                    resolve();
                }, 3000);
            });
        };
        
        const b = () => {
            console.log("b");
        };
        
        a().then(b);
   ```
   promise 반환 하는 함수은 then과 catch, finally를 이용하여 동기적인 처리를 할 수 있다.
   ```javascript
        const a = () => {
            return new Promise((resolve) => {
                setTimeout(() => {
                    console.log("a");
                    resolve();
                }, 3000);
            });
        };
        
        const b = () => {
            return new Promise((resolve) => {
                setTimeout(() => {
                    console.log("b");
                    resolve();
                }, 3000);
            });
        };
        
        const c = () => {
            console.log("c");
        };
        
        a()
        .then(b)
        .then(c)
        .catch(
            console.log("문제 발생")
        )
        .finally(() => {
            console.log("모든 함수 완료");
        });
   ```
   하지만 promise 역시 코드가 길어지면 프로미스 헬이라고 끝도 없이 복잡해진다.


3. async와 await   
   위의 두 가지 문제를 해결하기 위해 만들어진 것으로 비동기 함수의 앞에 await를 붙이면
   해당 함수가 promise 객체를 반환하기 전까지 다음 함수로 넘어가지 않는다.
   단, await를 쓴 함수가 내부에 있을 경우 상위 함수의 앞에 async를 붙여 처리해야한다.
   ```javascript
        const a = () => {
            return new Promise((resolve) => {
                setTimeout(() => {
                    console.log("a");
                    resolve();
                }, 3000);
            });
        };
        
        const b = () => {
            return new Promise((resolve) => {
                setTimeout(() => {
                    console.log("b");
                    resolve();
                }, 3000);
            });
        };
        
        const c = () => {
            console.log("c");
        };
        
        const runAll = async () => {
            await a();
            await b();
            c();
        };
        
        runAll();
   ```

이렇게 세 가지 방법으로 동기 처리를 할 수 있다. 단순히 여러개의 함수를 나열하는 방식이라고 한다면
이렇게 처리해도 문제가 없다. 하지만 만약에 반복문 안에서 이러한 오래 걸리는 함수를 구동시킨다면 어떻게 될까?
다음의 코드를 보자

```javascript
   const delay = () => {
      const randomDelay = Math.floor(Math.random() * 4) * 100
      return new Promise(resolve => setTimeout(resolve, randomDelay))
   }
   
   const list = [1, 2, 3, 4, 5, 6, 7]
   
   for (let i=0;i<list.length;i++) {
      delay().then(() => console.log(list[i]))
   }
   // 출처 : https://tecoble.techcourse.co.kr/post/2020-09-01-loop-async/
```
딜레이 함수는 좋은게 있어서 갖고 왔다. 일반적으로 이 딜레이 함수는 파일 작업이거나 api 요청일 것이다.
이러한 코드일때 구동시켜보면 당연하지만 비동기 문제가 발생해서 순서대로 출력되지 않는다.
이걸 동기적으로 처리하기 위해서는 어떻게 하느냐? await이나 async를 달아서 처리해야한다.
문제는 await를 쓰려면 상위 함수에 async를 써야하기 때문에 함수로 한번 둘러싸야한다.

```javascript
   const delay = () => {
      const randomDelay = Math.floor(Math.random() * 4) * 100
      return new Promise(resolve => setTimeout(resolve, randomDelay))
   }
   
   const result = async (list) => {
      for (let i =0;i<list.length;i++) {
         await delay()
                 .then(() => console.log(list[i]))
      }
   }

   // const result = async (list) => {
   //    for (const data of list) {
   //       await delay()
   //               .then(() => console.log(data))
   //    }
   // }
   
   const list = [1, 2, 3, 4, 5, 6, 7]
   
   result(list)
```
이렇게 함수로 감싸서 for 처리를 한다면 동기적으로 돌아간다. 혹은 주석 처리한 것 같이 for of 반복문을 써도 된다.
하지만 이것은 안된다.

```javascript
   const delay = () => {
      const randomDelay = Math.floor(Math.random() * 4) * 100
      return new Promise(resolve => setTimeout(resolve, randomDelay))
   }
   
   let list = [1, 2, 3, 4, 5]
   
   list.forEach(async data => {
      await delay().then(() => console.log(data))
   })
```

이 경우 순서대로 출력이 안된다. 즉 동기적으로 처리되지 않는다. async await을 썼는데도 비동기라고 싶을 수 있지만
해당 forEach안에서만 동기적으로 처리되는거지 forEach에 대해서 동기적으로 처리되는게 아니다.
이러한 경우는 동기적으로 처리가 되지 않으며 list 안의 값이 무작위로 출력됨을 알 수 있다.

정리하자면 비동기로 돌아갈 것들을 동기적으로 반복 처리되기 위해서는 다음과 같다.
1. 함수 덧 씌워서 async 처리를 한다.
2. for-loop, for of loop를 사용한다.
3. 대상 함수를 await처리를 하거나 promise를 반환하게 한다.

위 세가지 조건을 충족하면 동기적인 반복문 처리가 된다.
