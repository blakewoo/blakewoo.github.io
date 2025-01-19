---
title: OPENSTACK - Horizon
author: blakewoo
date: 2025-1-19 17:00:00 +0900
categories: [Openstack]
tags: [Openstack, horizon] 
render_with_liquid: false
---

# Horizon

## 1. 개요
Openstack에서 공식적으로 제공하는 Dashboard 프로젝트이다.   
nova, cinder등 여러 프로젝트들을 웹 기반으로 손 쉽게 사용할 수 있는 인터페이스를 제공한다.

## 2. 구성
Horizon 프로젝트는 다른 Openstack과는 다르게 큰 두 개의 다른 컴포넌트로 이루어져있다.
하나는 Horizon일고 하나는 openstack_dashboard이다.

Horizon 디렉토리는 Django 프로젝트에서 사용하는 일반적인 라이브러리와 컴포넌트들이 있고
openstack_dashboard 디렉토리에는 호라이즌에서 사용하는 Django 프로젝트 참조들이 포함되어있다.

라고 공식 documentation 에는 적혀있지만 코드를 까보면 openstack_auth 라는 컴포넌트가 하나 더 있다.
이는 Keystone이랑 연동하기 위한 코드들도 이루어져있다.

## 3. 다른 프로젝트와 연동
아래 내용은 Horizon 공식 Document에 포함된 내용을 번역해서 갖고 온 것인데 이 부분은 Devstack으로 실질적인
검증에 들어가서 추가적으로 포스팅 해볼 예정이다.

### 1) 대시보드 구성

#### a. **새 대시보드 추가하기**
1. **구성 파일 추가**
  - 프로젝트에 새 대시보드를 추가하려면,  
    `openstack_dashboard/local/enabled` 디렉토리에 구성 파일을 추가한다.
  - 구성 파일은 새 대시보드의 등록 및 설정을 담당한다.

#### b. **URL 설정**
2. **`urls.py` 업데이트**
  - 프로젝트의 `urls.py` 파일에 다음 한 줄을 추가하면 된다.

    ```python
    url(r'', include(horizon.urls)),
    ```  

  - 위의 설정은 등록된 Horizon 앱을 기반으로 URL을 자동으로 생성한다.
  - 만약 다른 URL 구조가 필요하다면, 직접 구성할 수도 있다.

### 2) 템플릿

#### a. **템플릿 태그를 활용한 내비게이션 생성**

- Horizon은 내비게이션을 자동으로 생성하기 위한 템플릿 태그를 제공한다.

##### b. **예제 1: nav.html**
```html
{% load horizon %}

<div class='nav'>
  {% horizon_main_nav %}
</div>
```

##### c. **예제 2: sidebar.html**
```html
{% load horizon %}

<div class='sidebar'>
  {% horizon_dashboard_nav %}
</div>
```

#### d. **동작 방식**
- 위 템플릿 태그는 현재 활성 상태의 대시보드 및 패널을 템플릿 컨텍스트 변수로 인식한다.
- 이를 통해 동적으로 적절한 내비게이션을 렌더링한다.

### 3) 앱 디자인
```
...
project/
|---__init__.py
|---dashboard.py <----- Horizon에 앱을 등록하고 대시보드 속성을 설정한다
|---overview/
|---images/
    |-- images
    |-- __init__.py
    |---panel.py <----- 앱에 패널을 등록하고 패널 속성을 정의한다.
    |-- snapshots/
    |-- templates/
    |-- tests.py
    |-- urls.py
    |-- views.py
...
...
```

#### a. 대시보드 클래스
`dashboard.py` 파일 내에 클래스 정의와 등록 프로세스를 포함한다.

```python
import horizon

# ObjectStorePanels는 PanelGroup의 예
# 패널 클래스에 대한 일반적인 내용은 아래를 참고
class ObjectStorePanels(horizon.PanelGroup):
    slug = "object_store"
    name = _("Object Store")
    panels = ('containers',)

class Project(horizon.Dashboard):
    name = _("Project")  # 탐색 메뉴에 표시
    slug = "project"     # URL에 표시
    # 패널은 문자열이나 클래스 참조일 수 있다. 예: ObjectStorePanels
    panels = (BasePanels, NetworkPanels, ObjectStorePanels)
    default_panel = 'overview'
    ...

horizon.register(Project)
```

#### b. 패널 클래스
`panel.py` 파일에서 패널 클래스를 정의하고 대시보드 클래스에 연결한다.

```python
import horizon
from openstack_dashboard.dashboards.project import dashboard

class Images(horizon.Panel):
    name = "Images"
    slug = 'images'
    permissions = ('openstack.roles.admin', 'openstack.service.image')
    policy_rules = (('endpoint', 'endpoint:rule'),)

# 다른 애플리케이션의 대시보드에 패널을 등록할 수 있다.
dashboard.Project.register(Images)
```

기본적으로 Panel 클래스는 동일 디렉토리에 있는 `urls.py` 파일을 찾아 URL 패턴을 패널에서 대시보드로,
그리고 Horizon으로 통합하여 전체적으로 확장 가능하고 구성 가능한 URL 구조를 만드는 구조이다.

정책 규칙은 `horizon/openstack_dashboard/conf/`에 정의된다. 권한은 Keystone에서 상속되며,
`openstack.roles.role_name` 또는 `openstack.services.service_name` 형식으로 표현된다.
이는 Keystone의 사용자 역할과 서비스 카탈로그 내 서비스와 관련된다.


# 참고문헌
- [오픈스택 - Horizon](https://docs.openstack.org/horizon/latest/)
- [오픈스택 - Horizon Quickstart](https://docs.openstack.org/horizon/latest/contributor/quickstart.html#quickstart)
