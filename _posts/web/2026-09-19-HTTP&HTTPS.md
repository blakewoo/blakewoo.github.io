---
title: HTTP와 HTTPS 구조
author: blakewoo
date: 2026-09-21 22:00:00 +0900
categories: [web]
tags: [web, http, https]
render_with_liquid: false
---

# HTTP와 HTTPS 구조

원래는 네트워크 쪽에 추가되야할 내용인데, 좀 더 찾기 쉽게 웹 카테고리에 올리려고 한다.   
HTTP나 HTTPS는 많이 들어봤지만 설명하라고 하면 굉장히 말이 궁해지는 경우가 많다. 때문에 이번 기회에 정리해두려고 한다.

## 1. 개요
앞서 네트워크 포스팅에서 설명했듯이 HTTP와 HTTPS는 응용 계층 프로토콜이다. 이러한 HTTP와 HTTPS 프로토콜의 스택을 그리면 아래와 같다.

![img.png](/assets\blog\web\http&https\img.png)

기본적으로 TCP/IP 위에서 구동되는데 HTTP의 경우에는 TCP/IP 위에서 구동되고 HTTPS는 TLS 위에서 구동된다.   
위의 그림으로도 어느정도 이해는 가겠지만, 좀 더 세부적으로 알아보도록 하자.

## 2. HTTP
![img_1.png](/assets/blog/web/http&https/img_1.png)

큰 그림은 위의 그림과 같다.   
기본적으로 IP위에 TCP위에 HTTP가 올라가는거라 TCP의 신뢰성 있는 전송은 기본적으로 깔린 방식이다.    
접근 하고자하는 서버가 어떤 IP를 가지고 있는지는 DNS SERVER를 통해 IP를 조회하게 된다.   
그 절차는 [이곳](https://blakewoo.github.io/posts/OSI-7-Layers-%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC%EA%B3%84%EC%B8%B5/)
을 참고하면 된다. 이후 IP를 얻은 뒤 해당 IP로 TCP 연결을 수립한다. 3-way handshake라고 불리는 이 방식은 [OSI 7 Layer - 전송계층](https://blakewoo.github.io/posts/OSI-7-Layers-%EC%A0%84%EC%86%A1%EA%B3%84%EC%B8%B5/)
포스팅에서 잘 설명하고 있으니 참고하면 좋다.

연결이 맺어지면 첫번째로 OPTION이라는 메소드를 이용해서 해당요청에 대해 어떤 메소드를 사용할 수 있는지 받아온다.
그 이후로는 OPTION에서 받아온 가능 한 메소드 리스트 중에 가능한 메소드로 특정 리소스에 요청을 보낼 수 있다.
이 메소드는 REST API 포스팅을 참고하라.

기본적으로 HTTP 요청은 아래와 같은 구조를 가진다.

![img.png](/assets/blog/web/http&https/img_2.png)

위 그림과 같이 시작 라인과 헤더, 바디로 이루어져있다.
여기서 Body는 POST, PATCH, PUT에서 사용하며 Body는 어떤 타입으로 보내는지 헤더에서 지정한다.
어떤 타입인지는 헤더에 Content-Type로 지정하며 종류는 아래와 같다.

| Content-Type                        | 용도            | 대표 사용 예        |
| ----------------------------------- | ------------- | -------------- |
| `application/json`                  | JSON 데이터      | REST API       |
| `application/xml`                   | XML 데이터       | XML API        |
| `application/pdf`                   | PDF 파일        | PDF 다운로드       |
| `application/octet-stream`          | 일반 바이너리 데이터   | 파일 다운로드        |
| `application/x-www-form-urlencoded` | HTML Form 데이터 | 일반 form submit |
| `multipart/form-data`               | 여러 필드 + 파일    | 파일 업로드         |
| `text/plain`                        | 일반 텍스트        | 문자열 응답         |
| `text/html`                         | HTML 문서       | 웹 페이지          |
| `text/css`                          | CSS           | 스타일시트          |
| `text/csv`                          | CSV 데이터       | CSV 파일         |
| `image/jpeg`                        | JPEG 이미지      | 이미지            |
| `image/png`                         | PNG 이미지       | 이미지            |
| `image/webp`                        | WebP 이미지      | 웹 이미지          |
| `image/gif`                         | GIF 이미지       | GIF            |
| `audio/mpeg`                        | MP3           | 오디오            |
| `video/mp4`                         | MP4           | 동영상            |

위와 같은 구조로 구성된 HTTP 요청은 TCP 패킷으로 나누어서 보내지는데, 최대 패킷 크기를 넘어서면 쪼개져서 전달되게 된다.

## 3. HTTPS 

![img_2.png](/assets/blog/web/http&https/img_3.png)

HTTPS의 큰 그림은 위와 같다. 기본적으로 DNS, IP, TCP 연결을 위한 Handshake까지는 동일하나, TLS 통신을 위한 핸드세이크가 추가되었다.   

> ※ 추가 업데이트 예정이다.
{: .prompt-tip }

# 참고문헌
- [TCP 공식 규격 문서](https://datatracker.ietf.org/doc/html/rfc9293)
- [HTTP 공통 공식 규격 문서](https://datatracker.ietf.org/doc/html/rfc9110)
