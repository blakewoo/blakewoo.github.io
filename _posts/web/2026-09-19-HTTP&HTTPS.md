---
title: HTTP와 HTTPS 구조
author: blakewoo
date: 2026-09-21 20:00:00 +0900
categories: [web]
tags: [web, http, https]
render_with_liquid: false
---

# HTTP와 HTTPS 구조

원래는 네트워크 쪽에 추가되야할 내용인데, 좀 더 찾기 쉽게 웹 카테고리에 올리려고 한다.   
HTTP나 HTTPS는 많이 들어봤지만 설명하라고 하면 굉장히 말이 궁해지는 경우가 많다. 때문에 이번 기회에 정리해두려고 한다.

## 1. 개요
앞서 네트워크 포스팅에서 설명했듯이 HTTP와 HTTPS는 응용 계층 프로토콜이다. 이러한 HTTP와 HTTPS 프로토콜의 스택을 그리면 아래와 같다.

![img.png](/assets/blog/web/http&https/img.png)

기본적으로 TCP/IP 위에서 구동되는데 HTTP의 경우에는 TCP/IP 위에서 구동되고 HTTPS는 TLS 위에서 구동된다.   
위의 그림으로도 어느정도 이해는 가겠지만, 좀 더 세부적으로 알아보도록 하자.

## 2. HTTP

![img_1.png](/assets/blog/web/http&https/img_1.png)

## 3. HTTPS 

![img_2.png](/assets/blog/web/http&https/img_2.png)

> ※ 추가 업데이트 예정이다.
{: .prompt-tip }

# 참고문헌
- 
