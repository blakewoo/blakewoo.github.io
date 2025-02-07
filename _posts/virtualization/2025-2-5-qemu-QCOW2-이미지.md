---
title: QEMU - QCOW2 이미지
author: blakewoo
date: 2025-2-6 15:30:00 +0900
categories: [Virtualization]
tags: [Virtualization, QEMU, QCOW2] 
render_with_liquid: false
---

# QCOW2 이미지
QCOW2(QEMU copy on write)는 가상 디스크 이미지의 저장 형식이다.   
(이외에도 RAW 형식이나 여타 다른 방식도 많다)

## 1. 구조

<table width="209">
<tbody>
<tr>
<td style="text-align: center; width: 200.333px;"><strong>QCOW2 Header</strong></td>
</tr>
<tr>
<td style="text-align: center; width: 200.333px;"><strong>Refcount Table</strong></td>
</tr>
<tr>
<td style="text-align: center; width: 200.333px;"><strong>Refcount Block</strong></td>
</tr>
<tr>
<td style="text-align: center; width: 200.333px;"><strong>L1 Table</strong></td>
</tr>
<tr>
<td style="text-align: center; width: 200.333px;"><strong>L2 Table</strong></td>
</tr>
<tr>
<td style="text-align: center; width: 200.333px;"><strong>Data Cluster</strong></td>
</tr>
<tr>
<td style="text-align: center; width: 200.333px;"><strong>L2 Table</strong></td>
</tr>
<tr>
<td style="text-align: center; width: 200.333px;"><strong>Data Cluster</strong></td>
</tr>
<tr>
<td style="text-align: center; width: 200.333px;"><strong>Data Cluster</strong></td>
</tr>
<tr>
<td style="text-align: center; width: 200.333px;"><strong>Data Cluster</strong></td>
</tr>
<tr>
<td style="text-align: center; width: 200.333px;"><strong>...</strong></td>
</tr>
</tbody>
</table>

별도의 스냅샷이 없다면 QCOW2는 위와 같은 이미지 구조를 따른다.    
위와 같은 구조는 아래와 같은 형태이다.

![img.png](/assets/blog/virtualization/qemu/qcow2/img.png)

기본적으로 Data cluster는 가상 디스크를 일정한 사이즈로 쪼개놓은 것이다.
이러한 cluster를 찾기 위해 2차 페이징과 비슷하게 l1, l2 table을 운용하여
Data cluster를 찾는 것이다.

### 1) QCOW2 Header
해당 이미지에 대해 정보를 담고 있는 Header이다.
이 헤더도 다음과 같이 이루어져있다.
기본적으로 Big Endian으로 표기되어있다.

<table style="width: 951.625px;">
<tbody>
<tr style="height: 26px;">
<td style="width: 89px; text-align: center; height: 26px;"><strong>Byte index</strong></td>
<td style="width: 151px; text-align: center; height: 26px;"><strong>이름</strong></td>
<td style="width: 688.625px; text-align: center; height: 26px;"><strong>내용</strong></td>
</tr>
<tr style="height: 26px;">
<td style="width: 89px; text-align: center; height: 26px;">0 ~ 3</td>
<td style="width: 151px; text-align: center; height: 26px;">magic</td>
<td style="width: 688.625px; text-align: center; height: 26px;">해당 파일이 QCOW임을 알리는 시그니처이다. QFI \xfb로 되어있다.</td>
</tr>
<tr style="height: 26px;">
<td style="width: 89px; text-align: center; height: 26px;">4 ~ 7</td>
<td style="width: 151px; text-align: center; height: 26px;">version</td>
<td style="width: 688.625px; text-align: center; height: 26px;">해당 QCOW2 이미지가 어떤 버전인지 명시한다. 2 또는 3이다.</td>
</tr>
<tr style="height: 65px;">
<td style="width: 89px; text-align: center; height: 65px;">8 ~ 15</td>
<td style="width: 151px; text-align: center; height: 65px;">backing_file_offset</td>
<td style="width: 688.625px; text-align: center; height: 65px;">
<p>backing file의 위치이다. 0이면 backing file 이 없는 것이다.</p>
</td>
</tr>
<tr style="height: 26px;">
<td style="width: 89px; text-align: center; height: 26px;">16 ~ 19</td>
<td style="width: 151px; text-align: center; height: 26px;">backing_file_size</td>
<td style="width: 688.625px; text-align: center; height: 26px;">backing file의 사이즈이다. 해당 파일이 없으면 정의되지 않는다.</td>
</tr>
<tr style="height: 51px;">
<td style="width: 89px; text-align: center; height: 51px;">20 ~ 23</td>
<td style="width: 151px; text-align: center; height: 51px;">cluster_bits</td>
<td style="width: 688.625px; text-align: center; height: 51px;">클러스터 내 오프셋을 주소 지정하는 데 사용되는 비트 수이다. 최소 9 이상이어야한다. 이는 클러스터 크기의 최소치가 512bytes이기 때문이다.</td>
</tr>
<tr style="height: 45.5278px;">
<td style="width: 89px; text-align: center; height: 45.5278px;">24 ~ 31</td>
<td style="width: 151px; text-align: center; height: 45.5278px;">size</td>
<td style="width: 688.625px; text-align: center; height: 45.5278px;">
<p>가상 디스크의 크기이다. 바이트 크기로 나타낸다</p>
<p>&nbsp;클러스터 크기가 2MB 일경우 최대 2EB(61비트), 512 Bytes의 경우 최대 128GB(37bits) 이다.</p>
</td>
</tr>
<tr style="height: 26px;">
<td style="width: 89px; text-align: center; height: 26px;">32 ~ 35</td>
<td style="width: 151px; text-align: center; height: 26px;">crypt_method</td>
<td style="width: 688.625px; text-align: center; height: 26px;"> 해당 가상 디스크의 암호화 여부로, 0이면 암호화되지 않은 것, 1이면 AES 암호화, 2이면 LUKS 암호화된 것이다.</td>
</tr>
<tr style="height: 26px;">
<td style="width: 89px; text-align: center; height: 26px;">36 ~ 39</td>
<td style="width: 151px; text-align: center; height: 26px;">l1_size</td>
<td style="width: 688.625px; text-align: center; height: 26px;">L1 Table의 크기로, Table 안에 몇 개의 항목이 있는지에 대한 값이다.</td>
</tr>
<tr style="height: 26px;">
<td style="width: 89px; text-align: center; height: 26px;">40 ~ 47</td>
<td style="width: 151px; text-align: center; height: 26px;">l1_table_offset</td>
<td style="width: 688.625px; text-align: center; height: 26px;">L1 Table의 시작 위치 주소다. cluster 경계와 맞춰져 있어야한다.</td>
</tr>
<tr style="height: 26px;">
<td style="width: 89px; text-align: center; height: 26px;">48 ~ 55</td>
<td style="width: 151px; text-align: center; height: 26px;">refcount_table_offset</td>
<td style="width: 688.625px; text-align: center; height: 26px;">refcount Table의 시작 주소 위치이다. 클러스터 경계와 맞춰져 있어야한다.</td>
</tr>
<tr style="height: 26px;">
<td style="width: 89px; text-align: center; height: 26px;">56 ~ 59</td>
<td style="width: 151px; text-align: center; height: 26px;">refcount_table_clusters</td>
<td style="width: 688.625px; text-align: center; height: 26px;">refcount Table이 갖고 있는 cluster의 개수이다.</td>
</tr>
<tr style="height: 26px;">
<td style="width: 89px; text-align: center; height: 26px;">60 ~ 63</td>
<td style="width: 151px; text-align: center; height: 26px;">nb_snapshots</td>
<td style="width: 688.625px; text-align: center; height: 26px;">이미지가 갖고 있는 스냅샷의 개수이다.&nbsp;</td>
</tr>
<tr style="height: 26px;">
<td style="width: 89px; text-align: center; height: 26px;">64 ~ 71</td>
<td style="width: 151px; text-align: center; height: 26px;">snapshots_offset</td>
<td style="width: 688.625px; text-align: center; height: 26px;">스냅샷의 시작 주소이며 클러스터 경계와 맞춰져 있어야한다.&nbsp;</td>
</tr>
</tbody>
</table>

### 2) Refcount Table
여러 refcount block의 위치를 관리하는 테이블이다.

### 3) Refcount Block
각 데이터 클러스터의 참조 횟수를 저장하는 블럭이다. 특정 클러스터가 몇개의 메타데이터나
스냅샷에서 참조되고 있는지 표기하는 것이다. 만약에 해당 값이 0개라면 아무곳에서도 사용되지
않는 클러스터이기 때문에 삭제 될 수 있다.

### 4) L1 Table
L2 Table을 찾는데 필요한 오프셋이 Table 형태로 나열되어있다.   

### 5) L2 Table
Data Cluster를 찾는데 필요한 오프셋이 Table 형태로 나열되어있다.

### 6) Data Cluster
실질적인 데이터를 담고 있는 Cluster이다.
Cluster는 최소 512Bytes 부터 최대 2MB까지 지정할 수 있는데
기본적으로는 64KB로 잡혀있으며 크게 잡을 수록 성능은 향상되지만 낭비되는 공간이 많다.

### 7) backing file
읽으려는 데이터가 cluster에 없을 경우 데이터를 갖고 오는 파일이다.   
클러스터와 동일한 사이즈와 포맷을 맞출 필요는 없으며 backing file은 그 자체의 backing file을 가질 수 있다.

### 8) Snapshot
가상 머신의 특정 상태를 저장하려고 할 때 찍어두는것이 스냅샷 기능이다.   
이는 원치 않는 작업이나 이전 상태로 돌아가고 싶을때 미리 저장해두는 것이다.

스냅샷을 활성화하면 L1 테이블을 복사한다. 이후 변경 사항은 원본과 다른 클러스터에 기재하여
원본은 유지하되 변경사항은 저장한다.


> ※ 본 포스팅은 추가적으로 업데이트 될 예정이다.
{: .prompt-tip }

# 참고문헌
- [QEMU 공식 도큐먼트 소개](https://www.qemu.org/docs/master/about/index.html)
- [Julio Faracco's blog - An Introduction to QCOW2 Image Format.](https://juliofaracco.wordpress.com/2015/02/19/an-introduction-to-qcow2-image-format/)
- [QEMU 공식 git - qcow2 document](https://github.com/qemu/qemu/blob/master/docs/interop/qcow2.txt)
- [Improving the performance of the qcow2 format - KVM Forum 2017](https://events.static.linuxfound.org/sites/events/files/slides/kvm-forum-2017-slides.pdf)
