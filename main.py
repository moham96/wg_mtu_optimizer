#!/usr/bin/env python3
"""
WireGuard MTU Finder - Find optimal MTU combinations for WireGuard peers
Based on: https://github.com/nitred/nr-wg-mtu-finder

This script tests various MTU combinations between WireGuard peers to find
the optimal settings for maximum throughput.

Usage:
    Server mode: python3 wg_mtu_finder.py --mode server
    Client mode: python3 wg_mtu_finder.py --mode client --server-ip <server_ip>
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import socket
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuration
@dataclass
class Config:
    # MTU ranges
    mtu_min: int = 1280
    mtu_max: int = 1500
    mtu_step: int = 20
    
    # Network settings
    server_port: int = 12345
    iperf3_port: int = 5201
    iperf3_duration: int = 10
    
    # Timing settings
    mtu_change_delay: int = 3
    test_timeout: int = 30
    ping_timeout: int = 5
    
    # Messaging settings
    udp_timeout: int = 10
    max_retries: int = 3
    
    # File settings
    log_level: str = "INFO"
    output_dir: str = "wg_mtu_results"
    csv_filename: str = "mtu_results.csv"
    heatmap_filename: str = "mtu_heatmap.png"
    
    # WireGuard interface
    wg_interface: str = "wg0"
    
    # Performance thresholds
    min_speed_mbps: float = 1.0  # Minimum acceptable speed
    max_latency_ms: float = 1000.0  # Maximum acceptable latency

# Message types for UDP communication
MESSAGE_TYPES = {
    "PING": 1,
    "PONG": 2,
    "SET_MTU": 3,
    "MTU_SET": 4,
    "START_TEST": 5,
    "TEST_COMPLETE": 6,
    "ERROR": 7,
    "TERMINATE": 8
}

@dataclass
class TestResult:
    server_mtu: int
    client_mtu: int
    upload_speed: float
    download_speed: float
    latency: float
    packet_loss: float
    timestamp: str
    success: bool
    error_message: str = ""

class MTUMessage:
    """Lightweight UDP message protocol for MTU coordination"""
    
    def __init__(self, msg_type: int, data: dict = None):
        self.msg_type = msg_type
        self.data = data or {}
        self.timestamp = time.time()
    
    def pack(self) -> bytes:
        """Pack message into bytes"""
        json_data = json.dumps(self.data).encode('utf-8')
        header = struct.pack('!IBd', self.msg_type, len(json_data), self.timestamp)
        return header + json_data
    
    @classmethod
    def unpack(cls, data: bytes) -> 'MTUMessage':
        """Unpack bytes into message"""
        if len(data) < 13:  # Minimum header size
            raise ValueError("Message too short")
        
        msg_type, data_len, timestamp = struct.unpack('!IBd', data[:13])
        json_data = data[13:13+data_len].decode('utf-8')
        parsed_data = json.loads(json_data)
        
        msg = cls(msg_type, parsed_data)
        msg.timestamp = timestamp
        return msg

class NetworkUtils:
    """Network utility functions"""
    
    @staticmethod
    def get_interface_mtu(interface: str) -> int:
        """Get current MTU of network interface"""
        try:
            result = subprocess.run(
                ["ip", "link", "show", interface],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split('\n'):
                if 'mtu' in line:
                    return int(line.split('mtu ')[1].split()[0])
        except Exception as e:
            logging.error(f"Error getting MTU for {interface}: {e}")
        return 1500  # Default MTU
    
    @staticmethod
    def set_interface_mtu(interface: str, mtu: int) -> bool:
        """Set MTU for network interface"""
        try:
            result = subprocess.run(
                ["sudo", "ip", "link", "set", "dev", interface, "mtu", str(mtu)],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            logging.error(f"Error setting MTU {mtu} for {interface}: {e}")
            return False

class IPerf3Runner:
    """IPerf3 test runner"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def run_server(self) -> subprocess.Popen:
        """Start iperf3 server"""
        cmd = [
            "iperf3", "-s", "-p", str(self.config.iperf3_port),
            "-1"  # Exit after one test
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    def run_client(self, server_ip: str, reverse: bool = False) -> Tuple[float, float]:
        """Run iperf3 client test"""
        cmd = [
            "iperf3", "-c", server_ip, "-p", str(self.config.iperf3_port),
            "-t", str(self.config.iperf3_duration),
            "-J"  # JSON output
        ]
        
        if reverse:
            cmd.append("-R")
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, 
                timeout=self.config.iperf3_duration + 10
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if 'end' in data and 'sum_received' in data['end']:
                    bps = data['end']['sum_received']['bits_per_second']
                    return bps / 1_000_000, 0  # Convert to Mbps
                
        except Exception as e:
            logging.error(f"IPerf3 client error: {e}")
        
        return 0.0, 0.0

class MTUFinder:
    """Main MTU finder class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.network_utils = NetworkUtils()
        self.iperf3 = IPerf3Runner(config)
        self.results: List[TestResult] = []

        # Create output directory
        Path(self.config.output_dir).mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        logging.debug(f"MTUFinder initialized with config: {self.config}")
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        log_file = Path(self.config.output_dir) / 'mtu_finder.log'
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.debug("Logging configured for MTUFinder")
    
    def ping_test(self, target_ip: str) -> Tuple[float, float]:
        """Perform ping test to measure latency and packet loss"""
        logging.debug(f"Starting ping test to {target_ip}")
        try:
            result = subprocess.run([
                "ping", "-c", "5", "-W", str(self.config.ping_timeout), target_ip
            ], capture_output=True, text=True, timeout=self.config.ping_timeout + 5)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                latency = 999.0
                loss = 100.0
                
                for line in lines:
                    if 'packet loss' in line:
                        try:
                            loss_str = line.split('%')[0].split()[-1]
                            loss = float(loss_str)
                            logging.debug(f"Parsed packet loss: {loss}%")
                        except (IndexError, ValueError):
                            logging.debug("Failed to parse packet loss line")
                    
                    if 'rtt min/avg/max' in line or ('avg' in line and 'ms' in line and '/' in line):
                        try:
                            if '=' in line:
                                stats_part = line.split('=')[1].strip()
                                stats_part = stats_part.replace('ms', '').strip()
                                values = stats_part.split('/')
                                if len(values) >= 2:
                                    latency = float(values[1])
                                    logging.debug(f"Parsed latency: {latency} ms")
                        except (IndexError, ValueError):
                            logging.debug("Failed to parse latency line")
                
                return latency, loss
        except Exception as e:
            logging.error(f"Ping test error: {e}")
        
        return 999.0, 100.0  # High latency, high loss on error

class MTUServer:
    """Server component for MTU testing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.finder = MTUFinder(config)
        self.running = False
        self.client_address = None
    
    async def start(self):
        """Start the MTU server"""
        logging.info(f"Starting MTU server on port {self.config.server_port}")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', self.config.server_port))
        sock.settimeout(1.0)
        
        self.running = True
        
        try:
            while self.running:
                try:
                    data, addr = sock.recvfrom(1024)
                    logging.debug(f"Received {len(data)} bytes from {addr}")
                    self.client_address = addr
                    
                    try:
                        message = MTUMessage.unpack(data)
                        logging.debug(f"Unpacked message: type={message.msg_type}, data={message.data}")
                        await self.handle_message(message, sock, addr)
                    except Exception as e:
                        logging.error(f"Error handling message: {e}")
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    logging.error(f"Server error: {e}")
                    break
        finally:
            logging.debug("Closing server socket")
            sock.close()
    
    async def handle_message(self, message: MTUMessage, sock: socket.socket, addr):
        """Handle incoming messages"""
        logging.debug(f"Handling message type: {message.msg_type} from {addr}")
        if message.msg_type == MESSAGE_TYPES["PING"]:
            response = MTUMessage(MESSAGE_TYPES["PONG"])
            logging.debug("Sending PONG response")
            sock.sendto(response.pack(), addr)
            
        elif message.msg_type == MESSAGE_TYPES["SET_MTU"]:
            mtu = message.data.get("mtu", 1500)
            logging.debug(f"Setting server MTU to {mtu}")
            success = self.finder.network_utils.set_interface_mtu(
                self.config.wg_interface, mtu
            )
            
            response = MTUMessage(
                MESSAGE_TYPES["MTU_SET"],
                {"success": success, "mtu": mtu}
            )
            logging.debug(f"Sending MTU_SET response: success={success}, mtu={mtu}")
            sock.sendto(response.pack(), addr)
            
            if success:
                logging.info(f"Server MTU set to {mtu}")
                time.sleep(self.config.mtu_change_delay)
            
        elif message.msg_type == MESSAGE_TYPES["START_TEST"]:
            # Start iperf3 server
            server_proc = self.finder.iperf3.run_server()
            
            response = MTUMessage(MESSAGE_TYPES["TEST_COMPLETE"])
            sock.sendto(response.pack(), addr)
            
            # Wait for iperf3 to complete
            server_proc.wait()
            
        elif message.msg_type == MESSAGE_TYPES["TERMINATE"]:
            logging.info("Received termination signal")
            self.running = False

class MTUClient:
    """Client component for MTU testing"""
    
    def __init__(self, config: Config, server_ip: str):
        self.config = config
        self.server_ip = server_ip
        self.finder = MTUFinder(config)
        self.sock = None
        logging.debug(f"MTUClient initialized for server {server_ip} with config: {self.config}")
    
    def connect(self) -> bool:
        """Connect to MTU server"""
        try:
            logging.debug("Creating UDP socket for client")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(self.config.udp_timeout)
            
            # Test connection with ping
            ping_msg = MTUMessage(MESSAGE_TYPES["PING"])
            logging.debug(f"Sending PING to server {self.server_ip}:{self.config.server_port}")
            self.sock.sendto(ping_msg.pack(), (self.server_ip, self.config.server_port))
            
            data, _ = self.sock.recvfrom(1024)
            response = MTUMessage.unpack(data)
            logging.debug(f"Received response to PING: {response.msg_type}")
            
            if response.msg_type == MESSAGE_TYPES["PONG"]:
                logging.info("Successfully connected to MTU server")
                return True
                
        except Exception as e:
            logging.error(f"Connection failed: {e}")
        
        return False
    
    def set_server_mtu(self, mtu: int) -> bool:
        """Set MTU on server"""
        msg = MTUMessage(MESSAGE_TYPES["SET_MTU"], {"mtu": mtu})
        logging.debug(f"Requesting server to set MTU to {mtu}")
        
        for retry in range(self.config.max_retries):
            try:
                self.sock.sendto(msg.pack(), (self.server_ip, self.config.server_port))
                logging.debug(f"Sent SET_MTU (attempt {retry+1})")
                data, _ = self.sock.recvfrom(1024)
                response = MTUMessage.unpack(data)
                logging.debug(f"Received MTU_SET response: {response.data}")
                
                if response.msg_type == MESSAGE_TYPES["MTU_SET"]:
                    return response.data.get("success", False)
                    
            except Exception as e:
                logging.warning(f"MTU set retry {retry + 1}: {e}")
                time.sleep(1)
        
        return False
    
    def run_test(self, server_mtu: int, client_mtu: int) -> TestResult:
        """Run a single MTU combination test"""
        timestamp = datetime.now().isoformat()
        
        logging.info(f"Testing MTU combination: Server={server_mtu}, Client={client_mtu}")
        
        # Set server MTU
        if not self.set_server_mtu(server_mtu):
            logging.error(f"Failed to set server MTU to {server_mtu}")
            return TestResult(
                server_mtu, client_mtu, 0, 0, 999, 100,
                timestamp, False, "Failed to set server MTU"
            )
        
        # Set client MTU
        if not self.finder.network_utils.set_interface_mtu(self.config.wg_interface, client_mtu):
            logging.error(f"Failed to set client MTU to {client_mtu}")
            return TestResult(
                server_mtu, client_mtu, 0, 0, 999, 100,
                timestamp, False, "Failed to set client MTU"
            )
        
        time.sleep(self.config.mtu_change_delay)
        
        # Test connectivity
        latency, packet_loss = self.finder.ping_test(self.server_ip)
        logging.debug(f"Ping test results: latency={latency}, packet_loss={packet_loss}")
        
        if latency > self.config.max_latency_ms or packet_loss > 50:
            logging.warning(f"High latency or packet loss: latency={latency}, loss={packet_loss}")
            return TestResult(
                server_mtu, client_mtu, 0, 0, latency, packet_loss,
                timestamp, False, "High latency or packet loss"
            )
        
        # Signal server to start iperf3
        start_msg = MTUMessage(MESSAGE_TYPES["START_TEST"])
        logging.debug("Sending START_TEST to server")
        self.sock.sendto(start_msg.pack(), (self.server_ip, self.config.server_port))
        
        # Wait for server to be ready
        time.sleep(2)
        
        # Run upload test
        logging.debug("Running upload test (client -> server)")
        upload_speed, _ = self.finder.iperf3.run_client(self.server_ip, reverse=False)
        time.sleep(1)
        
        # Run download test
        logging.debug("Running download test (server -> client)")
        download_speed, _ = self.finder.iperf3.run_client(self.server_ip, reverse=True)
        
        success = upload_speed > 0 and download_speed > 0
        logging.debug(f"Test result: upload={upload_speed}, download={download_speed}, success={success}")
        
        return TestResult(
            server_mtu, client_mtu, upload_speed, download_speed,
            latency, packet_loss, timestamp, success
        )
    
    def run_full_test(self):
        """Run full MTU test matrix"""
        if not self.connect():
            logging.error("Failed to connect to server")
            return
        
        # Generate MTU combinations
        mtu_values = list(range(
            self.config.mtu_min, 
            self.config.mtu_max + 1, 
            self.config.mtu_step
        ))
        
        total_tests = len(mtu_values) ** 2
        current_test = 0
        
        logging.info(f"Starting MTU test matrix: {len(mtu_values)}x{len(mtu_values)} = {total_tests} tests")
        
        # CSV file setup
        csv_path = Path(self.config.output_dir) / self.config.csv_filename
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Server_MTU', 'Client_MTU', 'Upload_Speed_Mbps', 
                'Download_Speed_Mbps', 'Latency_ms', 'Packet_Loss_%',
                'Timestamp', 'Success', 'Error_Message'
            ])
            
            for server_mtu in mtu_values:
                for client_mtu in mtu_values:
                    current_test += 1
                    
                    logging.info(f"Progress: {current_test}/{total_tests} ({100*current_test/total_tests:.1f}%)")
                    
                    result = self.run_test(server_mtu, client_mtu)
                    self.finder.results.append(result)
                    
                    # Write to CSV
                    writer.writerow([
                        result.server_mtu, result.client_mtu,
                        result.upload_speed, result.download_speed,
                        result.latency, result.packet_loss,
                        result.timestamp, result.success, result.error_message
                    ])
                    csvfile.flush()
                    
                    # Check if we should terminate due to poor performance
                    if (result.upload_speed < self.config.min_speed_mbps and 
                        result.download_speed < self.config.min_speed_mbps and
                        result.success):
                        logging.warning(f"Very low speeds detected at MTU {server_mtu}/{client_mtu}")
        
        # Generate reports
        self.generate_heatmap()
        self.generate_summary()
        
        # Send termination signal
        term_msg = MTUMessage(MESSAGE_TYPES["TERMINATE"])
        self.sock.sendto(term_msg.pack(), (self.server_ip, self.config.server_port))
        
        logging.info("MTU testing completed!")
    
    def generate_heatmap(self):
        """Generate heatmap visualization"""
        if not self.finder.results:
            return
        
        # Create DataFrame
        df = pd.DataFrame([asdict(r) for r in self.finder.results])
        
        # Create pivot tables for heatmaps
        upload_pivot = df.pivot(
            index='server_mtu', 
            columns='client_mtu', 
            values='upload_speed'
        )
        download_pivot = df.pivot(
            index='server_mtu', 
            columns='client_mtu', 
            values='download_speed'
        )
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Upload speed heatmap
        sns.heatmap(
            upload_pivot, annot=True, fmt='.1f', cmap='viridis',
            ax=ax1, cbar_kws={'label': 'Upload Speed (Mbps)'}
        )
        ax1.set_title('Upload Speed by MTU Combination')
        ax1.set_xlabel('Client MTU')
        ax1.set_ylabel('Server MTU')
        
        # Download speed heatmap
        sns.heatmap(
            download_pivot, annot=True, fmt='.1f', cmap='viridis',
            ax=ax2, cbar_kws={'label': 'Download Speed (Mbps)'}
        )
        ax2.set_title('Download Speed by MTU Combination')
        ax2.set_xlabel('Client MTU')
        ax2.set_ylabel('Server MTU')
        
        plt.tight_layout()
        plt.savefig(
            Path(self.config.output_dir) / self.config.heatmap_filename,
            dpi=300, bbox_inches='tight'
        )
        
        logging.info(f"Heatmap saved to {self.config.heatmap_filename}")
    
    def generate_summary(self):
        """Generate test summary"""
        if not self.finder.results:
            return
        
        df = pd.DataFrame([asdict(r) for r in self.finder.results])
        
        # Find best combinations
        best_upload = df.loc[df['upload_speed'].idxmax()]
        best_download = df.loc[df['download_speed'].idxmax()]
        best_combined = df.loc[(df['upload_speed'] + df['download_speed']).idxmax()]
        
        summary = f"""
WireGuard MTU Test Summary
=========================
Generated: {datetime.now().isoformat()}

Test Configuration:
- MTU Range: {self.config.mtu_min} - {self.config.mtu_max} (step: {self.config.mtu_step})
- Total Tests: {len(df)}
- Successful Tests: {len(df[df['success']])}
- Failed Tests: {len(df[~df['success']])}

Best Results:
- Best Upload: {best_upload['upload_speed']:.2f} Mbps at Server MTU {best_upload['server_mtu']}, Client MTU {best_upload['client_mtu']}
- Best Download: {best_download['download_speed']:.2f} Mbps at Server MTU {best_download['server_mtu']}, Client MTU {best_download['client_mtu']}
- Best Combined: {best_combined['upload_speed'] + best_combined['download_speed']:.2f} Mbps at Server MTU {best_combined['server_mtu']}, Client MTU {best_combined['client_mtu']}

Statistics:
- Average Upload Speed: {df['upload_speed'].mean():.2f} Mbps
- Average Download Speed: {df['download_speed'].mean():.2f} Mbps
- Average Latency: {df['latency'].mean():.2f} ms
- Average Packet Loss: {df['packet_loss'].mean():.2f}%
"""
        
        summary_path = Path(self.config.output_dir) / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(summary)
        logging.info(f"Summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='WireGuard MTU Finder')
    parser.add_argument('--mode', choices=['server', 'client'], required=True,
                        help='Run mode: server or client')
    parser.add_argument('--server-ip', help='Server IP address (required for client mode)')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--interface', default='wg0', help='WireGuard interface name')
    parser.add_argument('--port', type=int, default=12345, help='Communication port')
    parser.add_argument('--mtu-min', type=int, default=1280, help='Minimum MTU to test')
    parser.add_argument('--mtu-max', type=int, default=1500, help='Maximum MTU to test')
    parser.add_argument('--mtu-step', type=int, default=20, help='MTU step size')
    parser.add_argument('--duration', type=int, default=10, help='IPerf3 test duration')
    parser.add_argument('--output-dir', default='wg_mtu_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Load from config file if specified
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            logging.basicConfig(level=getattr(logging, config.log_level, "INFO"))
            logging.debug(f"Loaded configuration from {args.config}: {config_data}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Override with command line arguments
    # Only override if the user explicitly set the argument (i.e., not using the default)
    if 'interface' in args and args.interface != parser.get_default('interface'):
        config.wg_interface = args.interface
    if 'port' in args and args.port != parser.get_default('port'):
        config.server_port = args.port
    if 'mtu_min' in args and args.mtu_min != parser.get_default('mtu_min'):
        config.mtu_min = args.mtu_min
    if 'mtu_max' in args and args.mtu_max != parser.get_default('mtu_max'):
        config.mtu_max = args.mtu_max
    if 'mtu_step' in args and args.mtu_step != parser.get_default('mtu_step'):
        config.mtu_step = args.mtu_step
    if 'duration' in args and args.duration != parser.get_default('duration'):
        config.iperf3_duration = args.duration
    if 'output_dir' in args and args.output_dir != parser.get_default('output_dir'):
        config.output_dir = args.output_dir
    
    # Check requirements
    if args.mode == 'client' and not args.server_ip:
        print("Error: --server-ip is required for client mode")
        sys.exit(1)
    
    # Check if running as root
    if os.geteuid() != 0:
        print("Warning: This script requires root privileges to change MTU settings")
        print("Please run with sudo")
        sys.exit(1)
    
    try:
        if args.mode == 'server':
            server = MTUServer(config)
            asyncio.run(server.start())
        else:
            client = MTUClient(config, args.server_ip)
            client.run_full_test()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()