---
title: HTTP와 HTTPS 구조
author: blakewoo
date: 2026-09-24 22:00:00 +0900
categories: [web]
tags: [web, http, https]
render_with_liquid: false
---

# HTTP와 HTTPS 구조

원래는 네트워크 쪽에 추가되야할 내용인데, 좀 더 찾기 쉽게 웹 카테고리에 올리려고 한다.   
HTTP나 HTTPS는 많이 들어봤지만 설명하라고 하면 굉장히 말이 궁해지는 경우가 많다. 때문에 이번 기회에 정리해두려고 한다.

## 1. 개요
앞서 네트워크 포스팅에서 설명했듯이 HTTP와 HTTPS는 응용 계층 프로토콜이다. 이러한 HTTP와 HTTPS 프로토콜의 스택을 그리면 아래와 같다.

![img.png](\assets\blog\web\http&https\img.png)

기본적으로 TCP/IP 위에서 구동되는데 HTTP의 경우에는 TCP/IP 위에서 구동되고 HTTPS는 TLS 위에서 구동된다.   
위의 그림으로도 어느정도 이해는 가겠지만, 좀 더 세부적으로 알아보도록 하자.

## 2. HTTP
![img_1.png](\assets/blog/web/http&https/img_1.png)

큰 그림은 위의 그림과 같다.   
기본적으로 IP위에 TCP위에 HTTP가 올라가는거라 TCP의 신뢰성 있는 전송은 기본적으로 깔린 방식이다.    
접근 하고자하는 서버가 어떤 IP를 가지고 있는지는 DNS SERVER를 통해 IP를 조회하게 된다.   
그 절차는 [이곳](https://blakewoo.github.io/posts/OSI-7-Layers-%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC%EA%B3%84%EC%B8%B5/)
을 참고하면 된다. 이후 IP를 얻은 뒤 해당 IP로 TCP 연결을 수립한다. 3-way handshake라고 불리는 이 방식은 [OSI 7 Layer - 전송계층](https://blakewoo.github.io/posts/OSI-7-Layers-%EC%A0%84%EC%86%A1%EA%B3%84%EC%B8%B5/)
포스팅에서 잘 설명하고 있으니 참고하면 좋다.

연결이 맺어지면 첫번째로 OPTION이라는 메소드를 이용해서 해당요청에 대해 어떤 메소드를 사용할 수 있는지 받아온다.
그 이후로는 OPTION에서 받아온 가능 한 메소드 리스트 중에 가능한 메소드로 특정 리소스에 요청을 보낼 수 있다.
이 메소드는 [REST API 포스팅](https://blakewoo.github.io/posts/REST-API/) 을 참고하라.

기본적으로 HTTP 요청은 아래와 같은 구조를 가진다.

![img.png](\assets/blog/web/http&https/img_2.png)

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

위와 같은 구조로 구성된 HTTP 요청은 도메인은 ascii code로 전송되며, 뒤에 Path 부분은 UTF-8 + Percent Encoding을 처리되어 전송되며 
헤더 역시 ascii code로 전송되는데 부득이하게 영어가 아닌 경우, 별도의 인코딩 방식에 의해 처리된다.   
위에서 말한 HTTP Body는 그냥 바이트 배열인데, 앞서 언급한 Content-Type에 따라서 다르게 인코딩하여 전송할 수 있다.   

- ```Content-Type: application/json``` : UTF-8 + Percent Encoding
- ```application/x-www-form-urlencoded``` : UTF-8 + Percent Encoding
- ```multipart/form-data``` : raw binary

이후 TCP 패킷으로 나누어서 보내지는데, 최대 패킷 크기를 넘어서면 쪼개져서 전달되게 된다.

### 1) Puny code
도메인이 한글이면 어떻게 할까? 혹은 영어권이 아닌 다른 나라라면?
이 경우 puny code라는 알고리즘을 이용해서 ascii 문자열로 변경한다.   
이렇게 변경하면 모두 ascii 코드로 처리할 수 있기 때문에 편의성이 올라간다.   
브라우저에서는 변환된 문자열을 역으로 원래 문자열로 변환해서 보여주기 때문에 잘 드러나지 않는 부분이다.   
변환 방식은 아래와 같다.

"go한글.com" 이라는 도메인이 있다고 가정해보겠다. 이를 puny code 문자열 인코딩하면 아래와 같이 변한다.   

```xn--go-f12i831k.com```

먼저 영어 부분만 떼어 적는다 위 예시의 경우에는 go한글이니 go만 남는다.   
이후 하이픈(-)을 쓰고, 남은 한글의 위치와 내용을 [puny code 방식](https://www.rfc-editor.org/info/rfc3492/#section-4) 으로 인코딩하여 붙이고 뒤에 ```.com```을 붙이면 된다.

### 2) Content-encoding
Content-type 과 Content-encoding은 다르다. Content-type은 이 콘텐츠는 어떤 것인가? 라면 Content-encoding은 이 데이터가 어떤 추가 변환을 거쳤는가? 에
가깝다.
가령 아래의 경우를 보자.

```
Content-Type: application/json
Content-Encoding: gzip
```

위와 같이 되어있다면 이 Content는 json이지만 gzip으로 압축되어있다라고 알 수 있는 것이다.

## 3. HTTPS 

![img_2.png](\assets/blog/web/http&https/img_3.png)

HTTPS의 큰 그림은 위와 같다. 기본적으로 DNS, IP, TCP 연결을 위한 Handshake와 그 뒤에 HTTP과 같은 통신은 동일하나, TLS 통신을 위한 핸드세이크가 추가되었다. 
TCP 통신이 연결되면 아래의 절차에 따라 TLS가 이루어진다.

### 1) TLS 절차
#### a. ClientHello
클라이언트가 서버에 아래와 같은 정보를 보내서 어떻게 TLS를 맺을지 정한다.
이 정보들은 하나의 항목에 다수개를 보낼 수 도 있는데 가령 "cipher_suites"의 경우에는 가능한 암호화 방식 후보군들을 보내는 방식이다.

| 정보      | 질문                | ClientHello 항목                  |
| ------- | ----------------- | ------------------------------- |
| TLS 버전  | 어떤 TLS를 사용할까?     | `supported_versions`            |
| 대칭 암호   | 데이터를 어떻게 암호화할까?   | `cipher_suites`                 |
| 키 교환    | 공유 비밀을 어떻게 만들까?   | `supported_groups`, `key_share` |
| 인증      | 어떤 서명을 검증할 수 있나?  | `signature_algorithms`          |
| 서버 선택   | 어느 웹사이트에 접속하나?    | `server_name (SNI)`             |
| 응용 프로토콜 | HTTP/1.1? HTTP/2? | `ALPN`                          |
| 세션 재사용  | 이전 연결을 재사용할까?     | `pre_shared_key`                |
| 0-RTT   | 바로 데이터를 보낼까?      | `early_data`                    |

#### b. ServerHello
서버에서는 ClientHello에서 보낸 후보중 대체로 하나만 선택해서 보낸다.

| ServerHello 항목              | 의미                               | TLS 1.3에서의 역할            | 예                        |
| --------------------------- | -------------------------------- | ------------------------ | ------------------------ |
| `legacy_version`            | 과거 TLS 버전 필드                     | 호환성을 위해 항상 `0x0303`      | TLS 1.2 값                |
| `random`                    | 서버가 생성한 32바이트 값                  | 핸드셰이크 및 downgrade 보호에 사용 | 32-byte random           |
| `legacy_session_id_echo`    | ClientHello의 Session ID를 그대로 돌려줌 | 구형 middlebox 호환성         | Client 값 그대로             |
| `cipher_suite`              | 사용할 Cipher Suite                 | 클라이언트가 제시한 목록 중 하나 선택    | `TLS_AES_128_GCM_SHA256` |
| `legacy_compression_method` | 과거 Compression 방식                | TLS 1.3에서는 반드시 `0`       | `0x00`                   |
| `supported_versions`        | 실제 선택된 TLS 버전                    | TLS 1.3 선택을 명확히 표시       | `TLS 1.3`                |
| `key_share`                 | 서버의 Key Exchange 공개 값            | ECDHE Shared Secret 생성   | X25519 public key        |
| `pre_shared_key`            | 선택한 PSK 표시                       | Session Resumption 시 사용  | selected identity        |

#### c. 암호화
##### a) 초기 암호화
처음 ClientHello를 보낼때 클라이언트에서는 ECDHE 키 쌍을 만든다. 이 키 쌍은 Private와 Public으로 되어있는데
public만 서버로 보낸다.
서버측 역시 ECDHE 키 쌍을 만들면 public만 클라이언트로 보낸다.

이후 서로 갖고 있는 private과 받은 public 키 값으로 ECDH를 이용해서 공유 비밀키(Shared Secret Key)를 만든다.
서로 다른 값으로 계산하지만 결과는 동일한 값으로 나온다.

이후 이 키를 가지고 양방향 암호키로 쓰진 않고 HKDF라는 Key Derivation Function을 쓴다.
아까 ECDHE로 만든 공유 비밀키로 Client Handshake Traffic Secret, Server Handshake Traffic Secret를 만든다음에
Handshake Traffic Secret으로 다시 암호화키와 IV 값을 만든다.

이후 진행되는 값은 이 handshake traffic secret값으로 암호화되어 서로 전송한다.

##### b) 인증서 체크
서버가 자신의 인증서를 보내게 되는데 보통 Certificate Chain 구조이며, 아래의 내용을 확인한다.

```
① 신뢰 가능한 CA가 발급했는가?

② Certificate Chain이 유효한가?

③ 인증서가 만료되지 않았는가?

④ 내가 접속한 도메인과
   인증서의 이름이 일치하는가?

⑤ 인증서 사용 목적 등이 적절한가?
```

위와 같은 내용을 확인하는 절차를 ```http://www.abc.com```이라는 예시를 두고 설명해보겠다.   

```
사용자:
https://www.abc.com 접속
           │
           ▼
서버가 Certificate 전송
           │
           ▼
① 인증서 체인 구성
           │
           ▼
② 각 인증서의 CA 서명 검증
           │
           ▼
③ 유효기간 / CA 권한 / Key Usage 등 검증
           │
           ▼
④ Root CA가 내 Trust Store에 있는지 확인
           │
           ▼
⑤ SAN에 www.abc.com이 있는지 확인
           │
           ▼
⑥ 폐기(Revocation) 상태 등 추가 검사
           │
           ▼
⑦ CertificateVerify 검증
   "서버가 실제 Private Key를 가지고 있는가?"
           │
           ▼
⑧ Finished 검증
           │
           ▼
        TLS 연결 성공
```

###### A.서버가 Certificate Chain 전송
서버는 클라이언트에게 Leaf 인증서, Intermediate CA(Certificate Authority, 인증기관) 인증서를 보내는데 Root CA 인증서까지는 일반적으로 보내지 않는다.
클라이언트는 이미 자신의 Trust Store에 Root CA를 가지고 있기 때문이다.

###### B. 각 인증서의 CA 서명 검증
각 인증서 안에는 TBSCertificate, Signature 값이 들어있다.
하위 인증서 내용(TBSCertificate)으로부터 만든 해시값에 대해 상위 인증서 CA가 자신의 Private Key로 서명한 결과가 Signature인데
이러한 Signature이 확실한지 알아보기 위해서는 상위 인증서 CA의 Public Key로 해당 값을 돌렸을때 하위 인증서 내용으로
만든 해시값과 동일한 값이 나온다면 문제가 없는 것이다.

자세한 절차는 아래와 같다.

```
Leaf TBSCertificate
        │
        ▼
      Hash
        │
        ▼
       H1
```

시그니처값을 Intermediate CA의 Public key를 이용해서 검증한다.
```
Signature Value
        +
Intermediate CA Public Key
        +
TBSCertificate
        │
        ▼
 Signature Verification
        │
        ▼
    성공 / 실패
```

이런 방식을 Root CA까지 반복하면 인증서 체인이 모두 검증된다.

```
Leaf Certificate
      │
      │ Signature 검증
      │ using Intermediate Public Key
      ▼
Intermediate Certificate
      │
      │ Signature 검증
      │ using Root Public Key
      ▼
Root CA
      │
      ▼
Trust Store
```

##### c) 암호화 통신
인증서까지 모두 체크하면 Application Traffic Secret가 이전의 공유 비밀키에서 생성되며 클라이언트 -> 서버로 가는 메세지는 Server Application Key로
서버 -> 클라이언트로 가는 메세지는 Client Application Key로 암호화하여 전송하게 된다.   
이 경우 클라이언트와 서버는 양쪽 키를 모두 가지고 있으며, 대칭키 암호화이기 때문에 전송과 수신간에 각각 전용 키로 암복호화를 한다.

이후 Header와 Body를 모두 암호화하며 전송한다.   
만약 HTTP 요청이 아래와 같다고 가정하겠다.

```
POST /api/users HTTP/1.1
Host: example.com
Authorization: Bearer abc123
Content-Type: application/json

{
  "name": "Kim",
  "password": "1234"
}
```

위 내용이 통째로 암호화 되며 이 겉에 붙는 헤더만 평문이다.
즉, 암호화 안된 부분은 아래와 같고
```
Source IP
Destination IP

Source Port
Destination Port

패킷 크기
TLS Record 크기

통신 시간
통신량
```

암호화 된 부분은 아래와 같다.
```
HTTP URL path
HTTP Header
Cookie
Authorization
HTTP Body
```

전체적인 구조는 아래와 같다.
```
┌─────────────────────────────┐
│ Ethernet Header             │ ← 보임
├─────────────────────────────┤
│ IP Header                   │ ← 보임
│ Src IP / Dst IP             │
├─────────────────────────────┤
│ TCP Header                  │ ← 보임
│ Src Port / Dst Port         │
├─────────────────────────────┤
│ TLS Record Header           │ ← 보임
├─────────────────────────────┤
│ TLS encrypted_record        │ ← 암호화
│                             │
│   HTTP Header               │
│   HTTP Body                 │
│                             │
└─────────────────────────────┘
```

> ※ 추가 업데이트 예정이다.
{: .prompt-tip }

# 참고문헌
- [TCP 공식 규격 문서](https://datatracker.ietf.org/doc/html/rfc9293)
- [HTTP 공통 공식 규격 문서](https://datatracker.ietf.org/doc/html/rfc9110)
- [Puny code 처리법](https://www.rfc-editor.org/info/rfc3492/#section-4)
