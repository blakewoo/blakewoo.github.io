---
title: Blockchain - Ethereum
author: blakewoo
date: 2025-10-13 21:30:00 +0900
categories: [Blockchain]
tags: [Blockchain, Ethereum] 
render_with_liquid: false
use_math: true
---

# Ethereum
## 1. 개요
비트코인은 사토시 나카모토의 백서에서 시작되었지만 이더리움은 비탈릭 부테린(Vitalik Buterin)의 2013년 작성된 백서에서 시작된 프로젝트로 비트코인에 이은 2세대 암호화폐이다.
블록체인을 ‘화폐’뿐 아니라 다양한 프로그램(스마트 계약)을 돌리는 플랫폼으로 확장하는게 목표이다.

화폐 단위는 이더(ETH)이며 비트코인의 경우 1억 사토시가 1 비트코인이었던 것처럼 이더 역시 가장 작은 단위는 wei이며 100경 wei는 1 이더이다.
가장 많이 사용하는 단위는 Gwei이다. 이는 Giga wei의 약자로, Shannon이라고도 한다.
 
각 화폐단위에 따른 이름은 아래와 같다.

<table>
    <tr>
        <td>Value (in wei)</td>
        <td>Exponent</td>
        <td>Common name</td>
        <td>SI name </td>
    </tr>
    <tr>
        <td>1</td>
        <td>1</td>
        <td>wei</td>
        <td>Wei </td>
    </tr>
    <tr>
        <td>1,000</td>
        <td>103</td>
        <td>Babbage</td>
        <td>Kilowei or femtoether </td>
    </tr>
    <tr>
        <td>1,000,000</td>
        <td>106</td>
        <td>Lovelace</td>
        <td>Megawei or picoether </td>
    </tr>
    <tr>
        <td>1,000,000,000</td>
        <td>109</td>
        <td>Shannon</td>
        <td>Gigawei or nanoether </td>
    </tr>
    <tr>
        <td>1,000,000,000,000</td>
        <td>1012</td>
        <td>Szabo</td>
        <td>Microether or micro </td>
    </tr>
    <tr>
        <td>1,000,000,000,000,000</td>
        <td>1015</td>
        <td>Finney</td>
        <td>Milliether or milli </td>
    </tr>
    <tr>
        <td>1,000,000,000,000,000,000</td>
        <td>1018</td>
        <td>Ether</td>
        <td>Ether </td>
    </tr>
    <tr>
        <td>1,000,000,000,000,000,000,000</td>
        <td>1021</td>
        <td>Grand</td>
        <td>Kiloether </td>
    </tr>
    <tr>
        <td>1,000,000,000,000,000,000,000,000</td>
        <td>1024</td>
        <td></td>
        <td>Megaether </td>
    </tr>
</table>

비트코인의 경우 최대 발행량이 2100만개로 고정되어있고 발행량은 일정 주기마다 반감되지만 이더리움은 고정된 공급량 상한선이 없다.
다만 프로토콜 규칙에 따라 발행량이 결정되며, 최근 업그레이드를 통해 시간이 지남에 따라 공급량을 줄일 수 있는 메커니즘이 도입되었다. 
또한 거래 수수료의 일부를 소각하기 때문에 네트워크 활동이 활발할 때는 발행량보다 더 많은 ETH가 소각될 수 있어 , 해당 기간 동안 공급량이 디플레이션될 수 있다.

## 2. 발행
블록체인의 경우에는 거래의 기록을 중재하며 이를 네트워크에 인증받기 위해 논스값에 따른 해시함수 값을 찾는 것을 통해 해당 거래가 확정되었음을 증명함으로써
거래 수수료와 네트워크로부터의 신규 비트코인을 발행 받았다면 이더리움의 경우에는 검증자가 일정 ETH를 예치하고 검증자로써 활동한다.
ETH를 예치했다고 모두 검증자가 되는건 아니지만 기본적으로 ETH를 많이 예치할 수록 검증자로 선택될 확률이 높아지며 검증자로 선택될 경우
거래들에 대해서 수집하고 검증한 뒤 예치한 ETH에 따라 이자를 받는 식으로 신규 발행이 이루어진다.   
그렇다고 예치한 ETH에 따라 이자가 늘어나진 않는다. 2025년 10월 12일 기준 최소 32ETH를 예치해야 검증자가 될 수 있으며
이자 역시 32ETH를 기준으로 지급된다.

## 3. 지분 증명
이더리움은 지분 증명(PoS) 방식을 사용한다. 
이 모델에서 검증자는 새로운 블록을 제안하고 확인할 기회를 얻기 위해 자신이 가지고 있는 일정량의 ETH를 사용하지 못하게 잠그거나 예치한다.
검증자가 누가 될지 선택은 무작위로 이루어지지만, 스테이킹된 ETH의 양에 따라 선택될 확률이 높아지며, 부정직하게 행동하는 검증자는 지분을 잃을 위험이 있다.
이를 통해 이더리움은 경제적 완결성을 확보할 수 있으며, 완결된 블록은 대개 약 15분 이내에 되돌리기가 매우 어렵다.
또한 이더리움은 충분한 수의 검증자가 동의하면 체크포인트를 사용하여 블록을 되돌릴 수 없음으로 표시한다.

## 4. 스마트 컨트랙트
## 5. 토큰

> ※ 추가 업데이트 및 검증 예정이다.
{: .prompt-tip }

# 참고문헌
- [github - ethereumbook : 02intro.asciidoc](https://github.com/ethereumbook/ethereumbook/blob/develop/02intro.asciidoc)
- [이더리움 - 이더리움과 비트코인: 차이점은 무엇인가?](https://ethereum.org/ko/ethereum-vs-bitcoin/)
- [이더리움 - 이더리움이란?](https://ethereum.org/ko/what-is-ethereum/) 
