---
title: 컴퓨터 구조 - Power Consumption
author: blakewoo
date: 2025-3-11 09:00:00 +0900
categories: [Computer science]
tags: [Computer science, Power Consumption] 
render_with_liquid: false
use_math: true
---

# 파워 소모

## 1. 개요
CPU 즉, 프로세서에서 파워 소모는 매우 중요한 이슈이다.    
이는 23년도부터 프로세서 클럭 수가 올라가는 속도가 둔화된 이유와도 직결되기 때문이다.   
이는 파워소모가 동반하는 발열 탓이다. 이 발열을 여러가지로 우회해보려고 현재도 많이 노력하지만
본질적으로 열이 발생하는 문제이기 때문에 2025년 현재도 4GHz를 넘는 CPU가 잘 없고 NVIDIA의 Blackwell에서
문제를 일으키는 것이다.   
이 때문에 단일 프로세서의 클럭을 높이는 방향보다 프로세서의 개수를 늘이는 방향으로 컴퓨터가 발달 된 것이기도 하다.

그렇다면 여기서 말하는 파워 소모란 무엇을 말하는 것일까?

프로세서에서 말하는 파워 소모(Power Consumption)은 사실 상 두 개의 총합이다.

$$ Power Consumption = Power_{Dynamic} + Power_{Static} $$

여기서 $Power_{Static}$ 은 프로세서가 아무것도 하지 않아도 소모되는 전력, 즉 대기전력을 말하는 것이고
$Power_{Dynamic}$ 은 프로세서가 작동할 때 즉 clock이 edge에서 up down 할때 회로가 소모하는 전력을 말하는 것이다.

> 에너지와 파워에 대해서 이야기할때 Power는 소모량, Energy는 일을 할 수 있는 능력이다.
Power를 시간에 대해서 적분하면 Energy가 되고 Energy를 시간에 대해서 미분하면 Power가 된다.   
그외의 수식으로 표현하면 아래와 같다.
$ Energy_{dynamic} = CapacitiveLoad \times Voltage^{2} $
{: .prompt-tip }

$Power_{Dynamic}$ 에 대한 식은 아래와 같다.

$$Power_{Dynamic} = \frac{1}{2} \times CapacityLoad \times Voltage^{2} \times FrequencySwitched$$

위의 식을 해석하자면

```
트랜지스터의 용량 x 전압의 제곱 x 클럭수
```

정도로 해석할 수 있다.   
기본적으로 트랜지스터의 용량은 총 트랜지스터에 비례한다고 볼 수 있다.   
(물론 트랜지스터가 작아진다면 개당 용량은 작아지지만 트랜지스터끼리 연결하는 와이어의 저항이 높아져서 발열이 심해진다)

전압은 말 그대로 트랜지스터에 인가되는 전압이 맞다.   
이 전압이 줄어들면 회로의 속도가 느려진다. 때문에 클럭 수도 그에 비례하여 줄여야한다.   
(역으로 전압이 높아지면 회로 속도가 빨라진다. 이는 오버클럭의 원리이기도 하다)   
또한 아래에서 언급할 $Power_{static}$ 이 높아진다.   
이는 트랜지스터가 ON 되는 임계전압과 전력을 공급하는 서플라이의 최대 전압 격차가 줄어들기 때문이라고 한다.

클럭수는 말 그대로 프로세서의 클럭이라고 생각하면 된다.

그렇다면 $Power_{Static}$ 는 무엇인가?

$Power_{Static}$ 에 대한 식은 아래와 같다.

$$ Power_{Static} = Current_{static}\times Voltage $$

여기서 $Current_{static} $ 는 가만있으면 소모되는 누설 전류이고
Voltage는 그냥 전압을 말한다.

## 2. 예시 문제
만약 어떤 프로세서 A에 대해서 전압을 15% 줄였다고 해보자, 그리고 이에 따라 clock도 15% 줄였다고 할때   
$Power_{dynamic}$은 얼마나 줄였는지 계산하면 어떻게 될까?

$$ Power_{dynamic} = \frac{1}{2} \times CapacitivedLoad \times Voltage^{2} \times FrequencySwitched $$
$$= \frac{1}{2}\times 0.85\times CapactiveLoad \times (0.85 \times Voltage)^{2} \times FrequencySwitched$$   
$$= (0.85)^{3} \times OldPower_{dynamic} $$   
$$ \approx 0.6 \times OldPower_{dynamic} $$

이전에 비해 40%가 줄었음을 알 수 있다.

