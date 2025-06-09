# WireGuard MTU Finder - Setup and Usage Guide

## Overview

This script helps you find the optimal MTU (Maximum Transmission Unit) settings for WireGuard connections by testing various MTU combinations between server and client peers. It measures upload/download speeds using iperf3 and generates comprehensive reports including heatmaps and CSV data.

## Features

- Tests all combinations of server/client MTU values
- Uses lightweight UDP messaging for coordination
- Measures upload/download speeds with iperf3
- Generates visual heatmaps and CSV reports
- Configurable parameters with sane defaults
- Automatic termination on poor performance
- Comprehensive logging

## Prerequisites

### System Requirements
- Linux system with root access
- Python 3.7 or higher
- WireGuard interface configured and active

### Required Python Packages
```bash
pip3 install pandas matplotlib seaborn numpy
```

### Required System Tools
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install iperf3 iproute2 iputils-ping

# CentOS/RHEL/Fedora
sudo dnf install iperf3 iproute iputils

# Arch Linux
sudo pacman -S iperf3 iproute2 iputils
```

## Installation

1. **Download the script:**
```bash
wget https://raw.githubusercontent.com/your-repo/wg-mtu-finder/main/wg_mtu_finder.py
chmod +x wg_mtu_finder.py
```

2. **Verify dependencies:**
```bash
python3 -c "import pandas, matplotlib, seaborn, numpy; print('All dependencies installed')"
which iperf3 ping ip
```

## Usage

### Basic Usage

#### On the Server (WireGuard peer that will act as server):
```bash
sudo python3 wg_mtu_finder.py --mode server
```

#### On the Client (WireGuard peer that will act as client):
```bash
sudo python3 wg_mtu_finder.py --mode client --server-ip <SERVER_WIREGUARD_IP>
```

### Advanced Usage

#### Server with custom settings:
```bash
sudo python3 wg_mtu_finder.py \
    --mode server \
    --interface wg0 \
    --port 12345 \
    --output-dir /tmp/mtu_results
```

#### Client with custom MTU range:
```bash
sudo python3 wg_mtu_finder.py \
    --mode client \
    --server-ip 10.0.0.1 \
    --interface wg0 \
    --mtu-min 1200 \
    --mtu-max 1500 \
    --mtu-step 50 \
    --duration 15
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | Required | `server` or `client` |
| `--server-ip` | Required for client | WireGuard IP of server peer |
| `--interface` | `wg0` | WireGuard interface name |
| `--port` | `12345` | UDP communication port |
| `--mtu-min` | `1280` | Minimum MTU to test |
| `--mtu-max` | `1500` | Maximum MTU to test |
| `--mtu-step` | `20` | MTU increment step |
| `--duration` | `10` | iperf3 test duration (seconds) |
| `--output-dir` | `wg_mtu_results` | Output directory |

## Step-by-Step Setup

### 1. Prepare WireGuard Environment

Ensure both peers have WireGuard configured and can communicate:

```bash
# Check WireGuard status
sudo wg show

# Test basic connectivity
ping <peer_wireguard_ip>
```

### 2. Firewall Configuration

Open required ports on both peers:

```bash
# Server peer - allow incoming connections
sudo ufw allow 12345/udp  # Communication port
sudo ufw allow 5201/tcp   # iperf3 port

# Client peer - allow outgoing (usually default)
# No additional rules needed unless restrictive outbound policy
```

### 3. Run the Test

#### Start Server First:
```bash
# On server peer
sudo python3 wg_mtu_finder.py --mode server --interface wg0
```

You should see:
```
2025-06-07 10:00:00,000 - INFO - Starting MTU server on port 12345
```

#### Start Client:
```bash
# On client peer  
sudo python3 wg_mtu_finder.py --mode client --server-ip 10.0.0.1 --interface wg0
```

### 4. Monitor Progress

The client will show progress updates:
```
2025-06-07 10:00:05,000 - INFO - Successfully connected to MTU server
2025-06-07 10:00:05,000 - INFO - Starting MTU test matrix: 12x12 = 144 tests
2025-06-07 10:00:06,000 - INFO - Testing MTU combination: Server=1280, Client=1280
2025-06-07 10:00:18,000 - INFO - Progress: 1/144 (0.7%)
2025-06-07 10:00:19,000 - INFO - Testing MTU combination: Server=1280, Client=1300
```

## Output Files

After completion, you'll find these files in the output directory:

### 1. CSV Results (`mtu_results.csv`)
```csv
Server_MTU,Client_MTU,Upload_Speed_Mbps,Download_Speed_Mbps,Latency_ms,Packet_Loss_%,Timestamp,Success,Error_Message
1280,1280,45.2,48.1,12.3,0.0,2025-06-07T10:00:18,True,
1280,1300,47.8,49.5,11.8,0.0,2025-06-07T10:00:35,True,
1300,1280,46.1,47.9,12.1,0.0,2025-06-07T10:00:52,True,
```

### 2. Visual Heatmap (`mtu_heatmap.png`)
- Side-by-side heatmaps showing upload and download speeds
- Color-coded performance matrix
- Easy identification of optimal MTU combinations

### 3. Summary Report (`summary.txt`)
```text
WireGuard MTU Test Summary
=========================
Generated: 2025-06-07T12:34:56

Test Configuration:
- MTU Range: 1280 - 1500 (step: 20)
- Total Tests: 144
- Successful Tests: 142
- Failed Tests: 2

Best Results:
- Best Upload: 52.3 Mbps at Server MTU 1420, Client MTU 1400
- Best Download: 54.1 Mbps at Server MTU 1400, Client MTU 1420
- Best Combined: 105.8 Mbps at Server MTU 1410, Client MTU 1410
```

### 4. Log File (`mtu_finder.log`)
Detailed execution log with timestamps and debug information.

## Interpreting Results

### 1. Optimal MTU Selection
- **Best Combined**: Use this for balanced upload/download performance
- **Asymmetric Usage**: Choose based on your primary traffic direction
- **Safety Margin**: Consider values slightly below optimal for stability

### 2. Heatmap Analysis
- **Dark colors**: Higher speeds (better performance)
- **Light colors**: Lower speeds (poor performance)
- **Patterns**: Look for consistent high-performance regions

### 3. Performance Considerations
- MTU values too high may cause fragmentation
- MTU values too low waste bandwidth on headers
- Network path MTU discovery issues may affect results

## Troubleshooting

### Common Issues

#### 1. Permission Errors
```bash
# Ensure running as root
sudo python3 wg_mtu_finder.py --mode server
```

#### 2. Connection Failures
```bash
# Check WireGuard connectivity
ping <peer_ip>

# Verify firewall rules
sudo ufw status
sudo iptables -L
```

#### 3. iperf3 Errors
```bash
# Test iperf3 manually
# On server:
iperf3 -s -p 5201

# On client:
iperf3 -c <server_ip> -p 5201
```

#### 4. MTU Setting Failures
```bash
# Check current MTU
ip link show wg0

# Test manual MTU change
sudo ip link set dev wg0 mtu 1400
```

### Debug Mode
Enable detailed logging:
```bash
# Modify script or add environment variable
export PYTHONPATH=/path/to/script
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# Then run your command
"
```

### Network Path Issues
If you experience inconsistent results:

1. **Check underlying network MTU:**
```bash
# Find path MTU to destination
tracepath <destination_ip>
```

2. **Test without WireGuard:**
```bash
# Temporarily test on physical interface
ping -M do -s 1472 <destination_ip>
```

## Advanced Configuration

### Custom Configuration File
Create `config.json`:
```json
{
    "mtu_min": 1200,
    "mtu_max": 1500,
    "mtu_step": 25,
    "iperf3_duration": 15,
    "test_timeout": 45,
    "min_speed_mbps": 5.0,
    "max_latency_ms": 100.0,
    "wg_interface": "wg0",
    "log_level": "DEBUG"
}
```

Use with:
```bash
sudo python3 wg_mtu_finder.py --mode client --server-ip 10.0.0.1 --config config.json
```

### Multiple Interface Testing
For multiple WireGuard interfaces:
```bash
# Test wg1 interface
sudo python3 wg_mtu_finder.py --mode server --interface wg1

# Test wg2 interface
sudo python3 wg_mtu_finder.py --mode client --server-ip 10.1.0.1 --interface wg2
```

### Batch Testing Script
Create `batch_test.sh`:
```bash
#!/bin/bash

INTERFACES=("wg0" "wg1" "wg2")
SERVER_IPS=("10.0.0.1" "10.1.0.1" "10.2.0.1")

for i in "${!INTERFACES[@]}"; do
    interface="${INTERFACES[$i]}"
    server_ip="${SERVER_IPS[$i]}"
    
    echo "Testing interface: $interface"
    sudo python3 wg_mtu_finder.py \
        --mode client \
        --server-ip "$server_ip" \
        --interface "$interface" \
        --output-dir "results_$interface"
done
```

## Performance Optimization Tips

### 1. Pre-filtering MTU Range
Run a quick test first to narrow the range:
```bash
# Quick test with larger steps
sudo python3 wg_mtu_finder.py \
    --mode client \
    --server-ip 10.0.0.1 \
    --mtu-min 1200 \
    --mtu-max 1500 \
    --mtu-step 100 \
    --duration 5
```

### 2. Parallel Testing
For multiple interface pairs, run tests in parallel on different machines.

### 3. Network Optimization
- Ensure minimal other network traffic during testing
- Use dedicated testing window during low-usage periods
- Consider CPU affinity for iperf3 processes

## Integration with WireGuard

### Applying Results
Once you find optimal MTU values:

1. **Update WireGuard configuration:**
```ini
[Interface]
# ... existing config ...
MTU = 1420

[Peer]
# ... existing config ...
```

2. **Restart WireGuard:**
```bash
sudo wg-quick down wg0
sudo wg-quick up wg0
```

3. **Verify applied settings:**
```bash
ip link show wg0
```

### Persistent Configuration
Add to systemd service or startup scripts:
```bash
# Add to /etc/systemd/system/wg-quick@wg0.service.d/override.conf
[Service]
ExecStartPost=/sbin/ip link set dev wg0 mtu 1420
```

## Automation and Monitoring

### Cron Job for Regular Testing
```bash
# Add to crontab (crontab -e)
0 2 * * 0 /usr/bin/sudo /usr/bin/python3 /opt/wg-mtu-finder/wg_mtu_finder.py --mode client --server-ip 10.0.0.1 --output-dir /var/log/wg-mtu/$(date +\%Y\%m\%d)
```

### Integration with Monitoring Systems
Export results to monitoring systems:
```bash
# Example: Send results to InfluxDB
python3 << EOF
import pandas as pd
from influxdb import InfluxDBClient

df = pd.read_csv('wg_mtu_results/mtu_results.csv')
best_result = df.loc[df['Upload_Speed_Mbps'].idxmax()]

client = InfluxDBClient('localhost', 8086, 'user', 'pass', 'wireguard')
point = {
    "measurement": "mtu_test",
    "fields": {
        "best_upload": float(best_result['Upload_Speed_Mbps']),
        "best_download": float(best_result['Download_Speed_Mbps']),
        "optimal_server_mtu": int(best_result['Server_MTU']),
        "optimal_client_mtu": int(best_result['Client_MTU'])
    }
}
client.write_points([point])
EOF
```

## Security Considerations

1. **Network Exposure**: The script opens UDP port 12345 temporarily
2. **Root Access**: Required for MTU changes - audit the script before use
3. **Traffic Analysis**: iperf3 generates significant traffic that could be monitored
4. **Firewall Rules**: Ensure test ports are closed after testing

## Support and Contributing

### Getting Help
- Check the log files for detailed error information
- Verify all prerequisites are installed
- Test basic WireGuard connectivity first
- Review firewall and network configuration

### Contributing
- Submit issues and feature requests on GitHub
- Include log files and system information
- Test thoroughly before submitting pull requests

### License
This script is provided under MIT License. Use at your own risk.

---

**Note**: Always test in a controlled environment before deploying optimal MTU settings in production. Network conditions can change, affecting optimal MTU values.