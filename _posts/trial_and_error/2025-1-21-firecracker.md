---
title: Firecracker 설치 및 간단한 운용
author: blakewoo
date: 2025-1-24 16:00:00 +0900
categories: [Trial and error]
tags: [Firecracker] 
render_with_liquid: false
---

# Firecracker

> ※ 본 포스팅은 2025년 1월 24일 기준으로 작성되었으며 포스팅 작성이 완료되지 않았다.
{: .prompt-tip }

## 1. 개요
aws에서 개발한 서버리스를 위한 가상화이다.   
세부적인 분석에 대한 포스팅은 추후할 예정이고 이번 포스팅에서는 firecracker를 직접 세팅해고자 한다.

먼저 설치 환경은 아래와 같다.
```
OS: UBUNTU 24.04.1 LTS
CPU: INTEL(R) CORE(TM) i7-8700K CPU @ 3.7GHZ 
RAM: 16GB (2113MHz)
STORAGE TYPE: SATA SSD 256GB
```

구동 이전에 kvm이 켜져있는지 확인해야한다.
firecracker는 kvm의 기능을 사용하기 때문에 해당 기능이 켜져있지 않으면 안되기 때문이다.

ubuntu의 경우 아래의 명령어를 입력하면 kvm을 지원하는지 알 수 있다.

```shell
lsmod | grep kvm
```

아래와 비슷하게 뜬다면 kvm을 사용할 수 있는 것이다.

```
kvm_intel             348160  0
kvm                   970752  1 kvm_intel
irqbypass              16384  1 kvm
```

아래부터는 편의상 root 계정에서 진행하도록 하겠다.

## 2. 설치 절차

실행하기 위해 두 가지 방법이 있다.
소스코드를 받아와서 빌드해서 사용하는 방법과 이미 빌드된 파일을 갖고 와서 구동하는 방법인데
일단 여기서는 빌드된 파일을 갖고 와서 구동하는 방법을 사용할 예정이다.

### 1) 이미지 다운로드 및 세팅

일단 먼저 게스트 커널 이미지가 필요하다.   
이 firecracker도 microVM이긴하지만 가상머신이긴 가상머신이기 때문이다.
친절하게 firecracker 공식 git에서는 이미지를 갖고오는 것에 대해 스크립트를 제공하고 있다.

먼저 아래와 같이 적합한 아키텍처의 이미지를 받아온다.
```shell
ARCH="$(uname -m)"

latest=$(wget "http://spec.ccfc.min.s3.amazonaws.com/?prefix=firecracker-ci/v1.11/$ARCH/vmlinux-5.10&list-type=2" -O - 2>/dev/null | grep -oP "(?<=<Key>)(firecracker-ci/v1.11/$ARCH/vmlinux-5\.10\.[0-9]{1,3})(?=</Key>)")

# Download a linux kernel binary
wget "https://s3.amazonaws.com/spec.ccfc.min/${latest}"
```

squashfs로 압축된 ubuntu 24.04 커널파일을 받아와서 압축 해제를 한다.

```shell
# Download a rootfs
wget -O ubuntu-24.04.squashfs.upstream "https://s3.amazonaws.com/spec.ccfc.min/firecracker-ci/v1.11/${ARCH}/ubuntu-24.04.squashfs"

# Create an ssh key for the rootfs
unsquashfs ubuntu-24.04.squashfs.upstream
```

이후 ssh-keygen으로 새로운 키를 만든 뒤에 받아온 커널 파일에 세팅하여 새로 올라간 vm을 ssh로 접근할 수 있게 키를 등록해준다.

```shell
ssh-keygen -f id_rsa -N ""
cp -v id_rsa.pub squashfs-root/root/.ssh/authorized_keys
mv -v id_rsa ./ubuntu-24.04.id_rsa
```

이미지에 권한을 부여하고 이미 만들어진 데이터를 ext4 방식으로 포맷한다.

```shell
# create ext4 filesystem image
sudo chown -R root:root squashfs-root
truncate -s 400M ubuntu-24.04.ext4
sudo mkfs.ext4 -d squashfs-root -F ubuntu-24.04.ext4
```

### 2) firecracker 빌드된 바이너리 가져오기

release된 tar로 압축된 파일을 공식 git에서 가져온다.
이후 압축 해제한 후에 안에 있는 binary 파일의 이름을 firecracker로 바꾼다.

```shell
ARCH="$(uname -m)"
release_url="https://github.com/firecracker-microvm/firecracker/releases"
latest=$(basename $(curl -fsSLI -o /dev/null -w  %{url_effective} ${release_url}/latest))
curl -L ${release_url}/download/${latest}/firecracker-${latest}-${ARCH}.tgz \
| tar -xz

# Rename the binary to "firecracker"
mv release-${latest}-$(uname -m)/firecracker-${latest}-${ARCH} firecracker
```

### 3) firecracker 구동 시키고 접속하기

여기서는 총 2개의 터미널이 필요하다.
firecracker를 구동할 터미널과 firecracker로 띄워진 microVM에 접속할 터미널이다.

#### a. firecracker 구동 터미널

이전에 관련 socket 통신이 있다면 끊어버리고 관련 파일을 삭제한 뒤에 새로 socker을 열어 firecracker를 구동한다.

```shell
API_SOCKET="/tmp/firecracker.socket"

# Remove API unix socket
sudo rm -f $API_SOCKET

# Run firecracker
sudo ./firecracker --api-sock "${API_SOCKET}"
```

#### b. firecracker microVM에 접속할 터미널

새로 만들어질 microVM의 이름은 tap0, ip는 172.16.0.1에 서브넷 마스크는 30bit까지 사용하는 것으로 네트워크 세팅을 해준다.

```shell
TAP_DEV="tap0"
TAP_IP="172.16.0.1"
MASK_SHORT="/30"

# Setup network interface
sudo ip link del "$TAP_DEV" 2> /dev/null || true
sudo ip tuntap add dev "$TAP_DEV" mode tap
sudo ip addr add "${TAP_IP}${MASK_SHORT}" dev "$TAP_DEV"
sudo ip link set dev "$TAP_DEV" up
```

ehco 명령어로 패킷포워딩 설정을 해준다.   
이 포워딩은 reboot하면 사라진다.

```shell
# Enable ip forwarding
sudo sh -c "echo 1 > /proc/sys/net/ipv4/ip_forward"
sudo iptables -P FORWARD ACCEPT
```

가상 인터페이스를 만들어준다.
```shell
# This tries to determine the name of the host network interface to forward
# VM's outbound network traffic through. If outbound traffic doesn't work,
# double check this returns the correct interface!
HOST_IFACE=$(ip -j route list default |jq -r '.[0].dev')

# Set up microVM internet access
sudo iptables -t nat -D POSTROUTING -o "$HOST_IFACE" -j MASQUERADE || true
sudo iptables -t nat -A POSTROUTING -o "$HOST_IFACE" -j MASQUERADE

API_SOCKET="/tmp/firecracker.socket"
```

생성하는 vm에 대한 log를 만들어준다.
```shell
LOGFILE="./firecracker.log"

# Create log file
touch $LOGFILE

# Set log file
sudo curl -X PUT --unix-socket "${API_SOCKET}" \
    --data "{
        \"log_path\": \"${LOGFILE}\",
        \"level\": \"Debug\",
        \"show_level\": true,
        \"show_log_origin\": true
    }" \
    "http://localhost/logger"
```

커널 구동을 위한 세팅이다.

```shell
KERNEL="./$(ls vmlinux* | tail -1)"
KERNEL_BOOT_ARGS="console=ttyS0 reboot=k panic=1 pci=off"

ARCH=$(uname -m)

if [ ${ARCH} = "aarch64" ]; then
    KERNEL_BOOT_ARGS="keep_bootcon ${KERNEL_BOOT_ARGS}"
fi

# Set boot source
sudo curl -X PUT --unix-socket "${API_SOCKET}" \
    --data "{
        \"kernel_image_path\": \"${KERNEL}\",
        \"boot_args\": \"${KERNEL_BOOT_ARGS}\"
    }" \
    "http://localhost/boot-source"

ROOTFS="./ubuntu-24.04.ext4"

# Set rootfs
sudo curl -X PUT --unix-socket "${API_SOCKET}" \
    --data "{
        \"drive_id\": \"rootfs\",
        \"path_on_host\": \"${ROOTFS}\",
        \"is_root_device\": true,
        \"is_read_only\": false
    }" \
    "http://localhost/drives/rootfs"
```

VM에 대한 MAC을 설정해준다.
```shell
# The IP address of a guest is derived from its MAC address with
# `fcnet-setup.sh`, this has been pre-configured in the guest rootfs. It is
# important that `TAP_IP` and `FC_MAC` match this.
FC_MAC="06:00:AC:10:00:02"

# Set network interface
sudo curl -X PUT --unix-socket "${API_SOCKET}" \
    --data "{
        \"iface_id\": \"net1\",
        \"guest_mac\": \"$FC_MAC\",
        \"host_dev_name\": \"$TAP_DEV\"
    }" \
    "http://localhost/network-interfaces/net1"

# API requests are handled asynchronously, it is important the configuration is
# set, before `InstanceStart`.
sleep 0.015s
```

microVm을 구동하고 미리 생성해둔 key file로 ssh를 통해 ip 세팅과 dns 세팅 후 접속한다.

```shell
# Start microVM
sudo curl -X PUT --unix-socket "${API_SOCKET}" \
    --data "{
        \"action_type\": \"InstanceStart\"
    }" \
    "http://localhost/actions"

# API requests are handled asynchronously, it is important the microVM has been
# started before we attempt to SSH into it.
sleep 2s

# Setup internet access in the guest
ssh -i ./ubuntu-24.04.id_rsa root@172.16.0.2  "ip route add default via 172.16.0.1 dev eth0"

# Setup DNS resolution in the guest
ssh -i ./ubuntu-24.04.id_rsa root@172.16.0.2  "echo 'nameserver 8.8.8.8' > /etc/resolv.conf"

# SSH into the microVM
ssh -i ./ubuntu-24.04.id_rsa root@172.16.0.2

# Use `root` for both the login and password.
# Run `reboot` to exit.
```




# 참고문헌
- [firecracker 공식 git](https://github.com/firecracker-microvm/firecracker)
- [firecracker quick start 가이드](https://github.com/firecracker-microvm/firecracker/blob/main/docs/getting-started.md)
