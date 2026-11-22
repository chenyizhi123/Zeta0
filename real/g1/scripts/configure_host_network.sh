#!/usr/bin/env bash
# Install the RTL8125 driver and configure the selected wired interface for G1.
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
task_package="$task_root/.local/downloads/realtek-r8125-dkms_9.018.00-1_amd64.deb"
if [[ $# -lt 2 || $# -gt 3 || ! $1 =~ ^[[:xdigit:]]{4}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\.[0-7]$ ]]; then
    echo "Usage: sudo bash scripts/configure_host_network.sh PCI_ADDRESS IPV4_ADDRESS/PREFIX [PROFILE_NAME]" >&2
    exit 2
fi
task_pci="/sys/bus/pci/devices/$1"
task_address=$(python3 -c 'from ipaddress import IPv4Interface; import sys; print(IPv4Interface(sys.argv[1]))' "$2")
task_profile=${3:-zeta0-g1}
if [[ $EUID -ne 0 ]]; then
    echo "需本机管理员认证。请加 sudo 后运行指定参数。" >&2
    exit 1
fi
[[ $(cat "$task_pci/vendor") == 0x10ec && $(cat "$task_pci/device") == 0x8125 ]]
python3 - <<'PY'
from pathlib import Path
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        names = {Path(s.decode(errors='replace')).name for s in p.read_bytes().split(b'\0') if s}
    except OSError:
        continue
    if names & {'g1_deploy_onnx_ref', 'record_episode.py'}:
        raise SystemExit('先正常退出 SONIC 身体控制与采集，再配置网卡。')
PY
echo 'c6f38b51d8ba532b59355841ce9ea2f937bf4683576420908a4eab35723a05cf  '"$task_package" | sha256sum -c -
# Use Ubuntu's DKMS rather than the unrelated version in the CUDA repository.
export CC=/usr/bin/gcc-12
[[ -f /lib/modules/$(uname -r)/build/Makefile ]]
task_packages=("$task_package")
for task_deb in \
    cpp-12_12.3.0-1ubuntu1~22.04.3_amd64.deb \
    dctrl-tools_2.24-3build2_amd64.deb dkms_2.8.7-2ubuntu2.2_all.deb \
    gcc-12_12.3.0-1ubuntu1~22.04.3_amd64.deb \
    libasan8_12.3.0-1ubuntu1~22.04.3_amd64.deb \
    libgcc-12-dev_12.3.0-1ubuntu1~22.04.3_amd64.deb \
    libtsan2_12.3.0-1ubuntu1~22.04.3_amd64.deb; do
    task_packages+=("$task_root/.local/downloads/$task_deb")
done
apt-get install --yes "${task_packages[@]}"
modprobe r8125
task_interface=''
for task_attempt in {1..40}; do
    for task_net in "$task_pci"/net/*; do
        [[ -e "$task_net" ]] || continue
        task_interface=$(basename "$task_net")
        [[ $task_interface != eth0 ]] && break
    done
    [[ -n $task_interface && $task_interface != eth0 ]] && break
    sleep .25
done
[[ -n $task_interface && $task_interface != eth0 ]]
if ! nmcli -g connection.id connection show "$task_profile" >/dev/null 2>&1; then
    nmcli connection add type ethernet ifname "$task_interface" con-name "$task_profile"
fi
nmcli connection modify "$task_profile" connection.interface-name "$task_interface" \
    connection.autoconnect yes connection.autoconnect-priority 100 \
    ipv4.method manual ipv4.addresses "$task_address" \
    ipv4.gateway '' ipv4.dns '' ipv4.never-default yes ipv6.method disabled
printf 'softdep r8169 pre: r8125\n' > /etc/modprobe.d/zeta0-r8125.conf
update-initramfs -u -k "$(uname -r)"
python3 - "$task_root/config/local.json" "$task_interface" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
config = json.loads(p.read_text())
config['robot_interface'] = sys.argv[2]
p.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
PY
if [[ $(cat "$task_pci/net/$task_interface/carrier" 2>/dev/null || true) == 1 ]]; then
    nmcli --wait 20 connection up "$task_profile" ifname "$task_interface"
else
    echo '有线驱动与静态地址已保存；插入网线后自动连接。'
fi
modinfo -F version r8125
ip -4 -brief address
echo "完成：$task_interface，G1 地址 $task_address；Wi-Fi 默认路由保留。"
