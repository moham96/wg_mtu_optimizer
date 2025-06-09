#!/usr/bin/env python3
"""
WireGuard MTU Finder - Advanced MTU optimization tool for WireGuard peers
Author: Generated for optimal WireGuard performance testing
Version: 2.0
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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Color and emoji constants
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class Emojis:
    ROCKET = "🚀"
    FIRE = "🔥"
    CHECK = "✅"
    CROSS = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    CHART = "📊" 
    CLOCK = "⏱️"
    NETWORK = "🌐"
    SPEED = "💨"
    ARROW_UP = "⬆️"
    ARROW_DOWN = "⬇️"
    GEAR = "⚙️"

# Lightweight messaging protocol using UDP
class MTUMessenger:
    """Lightweight UDP-based messaging for MTU coordination"""
    
    def __init__(self, port: int = 12345):
        self.port = port
        self.socket = None
        
    def start_server(self):
        """Start UDP server for receiving messages"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('', self.port))
        self.socket.settimeout(5.0)
        
    def send_message(self, host: str, message: str) -> bool:
        """Send message to peer"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(3.0)
            data = message.encode('utf-8')
            sock.sendto(data, (host, self.port))
            sock.close()
            return True
        except Exception as e:
            logging.error(f"Failed to send message: {e}")
            return False
            
    def receive_message(self) -> Optional[Tuple[str, str]]:
        """Receive message from peer"""
        try:
            data, addr = self.socket.recvfrom(1024)
            return data.decode('utf-8'), addr[0]
        except socket.timeout:
            return None
        except Exception as e:
            logging.error(f"Failed to receive message: {e}")
            return None
            
    def close(self):
        """Close socket"""
        if self.socket:
            self.socket.close()

class WireGuardMTUFinder:
    """Advanced WireGuard MTU finder with comprehensive testing capabilities"""
    
    def __init__(self, config: Dict):
        self.config = self._validate_config(config)
        self.results = defaultdict(dict)
        self.best_result = {'speed': 0, 'server_mtu': 0, 'peer_mtu': 0}
        self.messenger = MTUMessenger(self.config.get('messenger_port', 12345))
        self.setup_logging()
        
    def _validate_config(self, config: Dict) -> Dict:
        """Validate and sanitize configuration"""
        # Ensure MTU values are reasonable
        mtu_min = max(576, min(9000, config.get('mtu_min', 1280)))
        mtu_max = max(576, min(9000, config.get('mtu_max', 1500)))
        
        if mtu_min >= mtu_max:
            print(f"{Colors.YELLOW}{Emojis.WARNING} Invalid MTU range: min={mtu_min}, max={mtu_max}{Colors.END}")
            mtu_min = 1280
            mtu_max = 1500
            
        config['mtu_min'] = mtu_min
        config['mtu_max'] = mtu_max
        config['mtu_step'] = max(1, min(100, config.get('mtu_step', 20)))
        config['test_duration'] = max(5, min(300, config.get('test_duration', 10)))
        config['mtu_change_delay'] = max(1, min(30, config.get('mtu_change_delay', 2)))
        
        return config
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # File handler
        log_file = Path(self.config.get('log_dir', './logs')) / f"mtu_finder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[console_handler, file_handler]
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        
    def print_colored(self, message: str, color: str = Colors.WHITE, emoji: str = ""):
        """Print colored message with emoji"""
        print(f"{color}{emoji} {message}{Colors.END}")
        
    def print_header(self, title: str):
        """Print formatted header"""
        border = "=" * (len(title) + 4)
        self.print_colored(border, Colors.CYAN, Emojis.ROCKET)
        self.print_colored(f"  {title}  ", Colors.CYAN, Emojis.ROCKET)
        self.print_colored(border, Colors.CYAN, Emojis.ROCKET)
        
    def get_current_mtu(self, interface: str) -> int:
        """Get current MTU of interface"""
        try:
            result = subprocess.run(['ip', 'link', 'show', interface], 
                                  capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):
                if 'mtu' in line:
                    return int(line.split('mtu')[1].split()[0])
        except Exception as e:
            self.logger.error(f"Failed to get MTU for {interface}: {e}")
        return 1420  # Default WireGuard MTU
        
    def set_mtu(self, interface: str, mtu: int) -> bool:
        """Set MTU for interface"""
        try:
            subprocess.run(['sudo', 'ip', 'link', 'set', 'dev', interface, 'mtu', str(mtu)], 
                          check=True, capture_output=True)
            self.logger.info(f"Set MTU {mtu} on {interface}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to set MTU {mtu} on {interface}: {e}")
            return False
            
    def run_iperf3_test(self, server_ip: str, duration: int = 10, 
                       direction: str = 'download') -> Optional[float]:
        """Run iperf3 speed test"""
        try:
            if direction == 'upload':
                cmd = ['iperf3', '-c', server_ip, '-t', str(duration), '-R', '-J']
            else:
                cmd = ['iperf3', '-c', server_ip, '-t', str(duration), '-J']
                
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=duration + 10)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Get bits per second and convert to Mbps
                bps = data['end']['sum_received']['bits_per_second']
                mbps = bps / 1_000_000
                return mbps
            else:
                self.logger.error(f"iperf3 failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("iperf3 test timed out")
            return None
        except Exception as e:
            self.logger.error(f"iperf3 test failed: {e}")
            return None
            
    def test_mtu_combination(self, server_mtu: int, peer_mtu: int, 
                           peer_ip: str) -> Optional[Dict]:
        """Test a specific MTU combination"""
        interface = self.config.get('interface', 'wg0')
        
        # Set local MTU
        if not self.set_mту(interface, server_mtu):
            return None
            
        # Coordinate with peer to set their MTU
        if not self.coordinate_peer_mtu(peer_ip, peer_mtu):
            return None
            
        # Wait for MTU changes to take effect
        time.sleep(self.config.get('mtu_change_delay', 2))
        
        # Test connectivity with ping
        if not self.test_connectivity(peer_ip):
            self.print_colored(f"No connectivity with MTU {server_mtu}/{peer_mtu}", 
                             Colors.RED, Emojis.CROSS)
            return None
            
        # Run speed tests
        test_duration = self.config.get('test_duration', 10)
        download_speed = self.run_iperf3_test(peer_ip, test_duration, 'download')
        upload_speed = self.run_iperf3_test(peer_ip, test_duration, 'upload')
        
        if download_speed is None or upload_speed is None:
            return None
            
        result = {
            'server_mtu': server_mtu,
            'peer_mtu': peer_mtu,
            'download_speed': download_speed,
            'upload_speed': upload_speed,
            'avg_speed': (download_speed + upload_speed) / 2,
            'timestamp': datetime.now().isoformat()
        }
        
        # Compare with best result
        if result['avg_speed'] > self.best_result['speed']:
            improvement = result['avg_speed'] - self.best_result['speed']
            self.print_colored(f"New best! {result['avg_speed']:.2f} Mbps "
                             f"(+{improvement:.2f}) with MTU {server_mtu}/{peer_mtu}", 
                             Colors.GREEN, Emojis.FIRE)
            self.best_result = {
                'speed': result['avg_speed'],
                'server_mtu': server_mtu,
                'peer_mtu': peer_mtu
            }
        else:
            change = result['avg_speed'] - self.best_result['speed']
            color = Colors.YELLOW if change > -5 else Colors.RED
            emoji = Emojis.ARROW_DOWN if change < 0 else Emojis.ARROW_UP
            self.print_colored(f"Speed: {result['avg_speed']:.2f} Mbps "
                             f"({change:+.2f}) with MTU {server_mtu}/{peer_mtu}", 
                             color, emoji)
        
        return result
        
    def coordinate_peer_mtu(self, peer_ip: str, mtu: int) -> bool:
        """Coordinate MTU setting with peer"""
        message = f"SET_MTU:{mtu}"
        if self.messenger.send_message(peer_ip, message):
            # Wait for acknowledgment
            for _ in range(5):  # 5 second timeout
                response = self.messenger.receive_message()
                if response and response[0] == "MTU_SET":
                    return True
                time.sleep(1)
        return False
        
    def test_connectivity(self, peer_ip: str) -> bool:
        """Test basic connectivity with ping"""
        try:
            result = subprocess.run(['ping', '-c', '3', '-W', '2', peer_ip], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
            
    def generate_mtu_ranges(self) -> List[Tuple[int, int]]:
        """Generate MTU combinations to test"""
        mtu_min = self.config.get('mtu_min', 1280)
        mtu_max = self.config.get('mtu_max', 1500)
        mtu_step = self.config.get('mtu_step', 20)
        
        server_range = range(mtu_min, mtu_max + 1, mtu_step)
        peer_range = range(mtu_min, mtu_max + 1, mtu_step)
        
        combinations = [(s, p) for s in server_range for p in peer_range]
        
        if self.config.get('randomize_order', False):
            import random
            random.shuffle(combinations)
            
        return combinations
        
    def binary_search_mtu(self, peer_ip: str) -> Dict:
        """Binary search for optimal MTU using bisect algorithm"""
        self.print_header("Binary Search Mode - Finding Optimal MTU")
        
        # Start with equal MTUs for simplicity in binary search
        low_mtu = self.config.get('mtu_min', 1280)
        high_mtu = self.config.get('mtu_max', 1500)
        mtu_step = self.config.get('mtu_step', 20)
        best_speed = 0
        best_mtu = low_mtu
        
        self.print_colored(f"Searching MTU range: {low_mtu} - {high_mtu}", 
                         Colors.CYAN, Emojis.GEAR)
        
        while low_mtu <= high_mtu:
            mid_mtu = (low_mtu + high_mtu) // 2
            self.print_colored(f"Testing MTU: {mid_mtu}", Colors.BLUE, Emojis.CLOCK)
            
            result = self.test_mtu_combination(mid_mtu, mid_mtu, peer_ip)
            
            if result is None:
                self.print_colored(f"MTU {mid_mtu} failed, trying lower", 
                                 Colors.YELLOW, Emojis.WARNING)
                high_mtu = mid_mtu - mtu_step
                continue
                
            if result['avg_speed'] > best_speed:
                best_speed = result['avg_speed']
                best_mtu = mid_mtu
                self.print_colored(f"New best MTU: {mid_mtu} at {best_speed:.2f} Mbps", 
                                 Colors.GREEN, Emojis.ROCKET)
                
                # Try higher MTU
                low_mtu = mid_mtu + mtu_step
            else:
                # Try lower MTU
                high_mtu = mid_mtu - mtu_step
                
        # Fine-tune around the best MTU found
        return self.fine_tune_mtu(best_mtu, peer_ip)
        
    def fine_tune_mtu(self, base_mtu: int, peer_ip: str) -> Dict:
        """Fine-tune MTU around the best found value"""
        self.print_colored("Fine-tuning around optimal MTU...", 
                         Colors.CYAN, Emojis.GEAR)
        
        mtu_min = self.config.get('mtu_min', 1280)
        mtu_max = self.config.get('mtu_max', 1500)
        
        fine_range = range(max(mtu_min, base_mtu - 20),
                          min(mtu_max, base_mtu + 21), 4)
        
        best_result = None
        for mtu in fine_range:
            result = self.test_mtu_combination(mtu, mtu, peer_ip)
            if result and (best_result is None or 
                          result['avg_speed'] > best_result['avg_speed']):
                best_result = result
                
        return best_result or {'server_mtu': base_mtu, 'peer_mtu': base_mtu, 
                              'avg_speed': 0}
        
    def full_matrix_test(self, peer_ip: str) -> Dict:
        """Run full matrix test of all MTU combinations"""
        self.print_header("Full Matrix Test - Testing All Combinations")
        
        combinations = self.generate_mtu_ranges()
        total_tests = len(combinations)
        
        self.print_colored(f"Testing {total_tests} MTU combinations", 
                         Colors.CYAN, Emojis.CHART)
        
        for i, (server_mtu, peer_mtu) in enumerate(combinations):
            progress = (i + 1) / total_tests * 100
            self.print_colored(f"Progress: {progress:.1f}% ({i+1}/{total_tests}) "
                             f"- Testing {server_mtu}/{peer_mtu}", 
                             Colors.BLUE, Emojis.CLOCK)
            
            result = self.test_mtu_combination(server_mtu, peer_mtu, peer_ip)
            if result:
                self.results[server_mtu][peer_mtu] = result
            else:
                # Mark failed combinations
                self.results[server_mtu][peer_mtu] = {
                    'server_mtu': server_mtu,
                    'peer_mtu': peer_mtu,
                    'download_speed': 0,
                    'upload_speed': 0,
                    'avg_speed': 0,
                    'failed': True
                }
                
            # Check for timeout/lag conditions
            if self.should_terminate_due_to_lag():
                self.print_colored("Terminating due to excessive lag/timeouts", 
                                 Colors.RED, Emojis.WARNING)
                break
                
        return self.best_result
        
    def should_terminate_due_to_lag(self) -> bool:
        """Check if we should terminate due to excessive lag"""
        # Implement logic to detect if connection is too slow/laggy
        # This is a placeholder - you can implement more sophisticated detection
        return False
        
    def save_results_csv(self):
        """Save results to CSV file"""
        output_dir = Path(self.config.get('output_dir', './results'))
        output_dir.mkdir(exist_ok=True)
        
        csv_file = output_dir / f"mtu_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['server_mtu', 'peer_mtu', 'download_speed', 
                           'upload_speed', 'avg_speed', 'timestamp'])
            
            for server_mtu, peer_results in self.results.items():
                for peer_mtu, result in peer_results.items():
                    writer.writerow([
                        result['server_mtu'],
                        result['peer_mtu'], 
                        result['download_speed'],
                        result['upload_speed'],
                        result['avg_speed'],
                        result.get('timestamp', '')
                    ])
                    
        self.print_colored(f"Results saved to: {csv_file}", Colors.GREEN, Emojis.CHECK)
        return csv_file
        
    def generate_heatmap(self):
        """Generate heatmap visualization of results"""
        if not self.results:
            self.print_colored("No results to visualize", Colors.YELLOW, Emojis.WARNING)
            return
            
        # Prepare data for heatmap
        server_mtus = sorted(self.results.keys())
        peer_mtus = sorted(set(peer_mtu for peer_results in self.results.values() 
                              for peer_mtu in peer_results.keys()))
        
        # Create matrix
        matrix = np.zeros((len(server_mtus), len(peer_mtus)))
        
        for i, server_mtu in enumerate(server_mtus):
            for j, peer_mtu in enumerate(peer_mtus):
                if peer_mtu in self.results[server_mtu]:
                    matrix[i][j] = self.results[server_mtu][peer_mtu]['avg_speed']
                    
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, 
                   xticklabels=[str(mtu) for mtu in peer_mtus],
                   yticklabels=[str(mtu) for mtu in server_mtus],
                   annot=True, 
                   fmt='.1f',
                   cmap='viridis',
                   cbar_kws={'label': 'Average Speed (Mbps)'})
        
        plt.title('WireGuard MTU Performance Heatmap')
        plt.xlabel('Peer MTU')
        plt.ylabel('Server MTU')
        plt.tight_layout()
        
        # Save heatmap
        output_dir = Path(self.config.get('output_dir', './results'))
        heatmap_file = output_dir / f"mtu_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.print_colored(f"Heatmap saved to: {heatmap_file}", Colors.GREEN, Emojis.CHART)
        
    def run_server_mode(self):
        """Run in server mode - coordinate with peer"""
        self.print_header("Server Mode - Coordinating MTU Tests")
        
        self.messenger.start_server()
        peer_ip = self.config['peer_ip']
        
        try:
            if self.config.get('binary_search', False):
                best_result = self.binary_search_mtu(peer_ip)
            else:
                best_result = self.full_matrix_test(peer_ip)
                
            # Save results
            csv_file = self.save_results_csv()
            
            # Generate visualizations
            if self.config.get('generate_heatmap', True):
                self.generate_heatmap()
                
            # Print final results
            self.print_final_results(best_result)
            
        finally:
            self.messenger.close()
            
    def run_peer_mode(self):
        """Run in peer mode - respond to server coordination"""
        self.print_header("Peer Mode - Responding to Server")
        
        self.messenger.start_server()
        interface = self.config.get('interface', 'wg0')
        
        try:
            self.print_colored("Waiting for MTU coordination messages...", 
                             Colors.CYAN, Emojis.NETWORK)
            
            while True:
                message = self.messenger.receive_message()
                if message:
                    msg_content, sender_ip = message
                    
                    if msg_content.startswith("SET_MTU:"):
                        mtu = int(msg_content.split(":")[1])
                        self.print_colored(f"Setting MTU to {mtu}", 
                                         Colors.BLUE, Emojis.GEAR)
                        
                        if self.set_mtu(interface, mtu):
                            self.messenger.send_message(sender_ip, "MTU_SET")
                            self.print_colored(f"Confirmed MTU {mtu}", 
                                             Colors.GREEN, Emojis.CHECK)
                        else:
                            self.messenger.send_message(sender_ip, "MTU_FAILED")
                            
                    elif msg_content == "TERMINATE":
                        self.print_colored("Received termination signal", 
                                         Colors.YELLOW, Emojis.INFO)
                        break
                        
        except KeyboardInterrupt:
            self.print_colored("Peer mode interrupted", Colors.YELLOW, Emojis.INFO)
        finally:
            self.messenger.close()
            
    def print_final_results(self, best_result: Dict):
        """Print final results summary"""
        self.print_header("Final Results Summary")
        
        if best_result and best_result.get('speed', 0) > 0:
            self.print_colored(f"Optimal MTU Configuration:", Colors.GREEN, Emojis.ROCKET)
            self.print_colored(f"  Server MTU: {best_result['server_mtu']}", Colors.WHITE)
            self.print_colored(f"  Peer MTU: {best_result['peer_mtu']}", Colors.WHITE)
            self.print_colored(f"  Best Speed: {best_result['speed']:.2f} Mbps", 
                             Colors.GREEN, Emojis.SPEED)
        else:
            self.print_colored("No optimal configuration found", Colors.RED, Emojis.CROSS)
            
        self.print_colored(f"Log file: {self.log_file}", Colors.CYAN, Emojis.INFO)

def load_config(config_file: str) -> Dict:
    """Load configuration from file with defaults"""
    default_config = {
        "interface": "wg0",
        "peer_ip": "",
        "mtu_min": 1280,
        "mtu_max": 1500,
        "mtu_step": 20,
        "test_duration": 10,
        "mtu_change_delay": 2,
        "messenger_port": 12345,
        "log_level": "INFO",
        "log_dir": "./logs",
        "output_dir": "./results",
        "binary_search": False,
        "generate_heatmap": True,
        "randomize_order": False
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            # Merge user config with defaults
            default_config.update(user_config)
        except Exception as e:
            print(f"{Colors.YELLOW}{Emojis.WARNING} Error loading config file: {e}{Colors.END}")
            print(f"{Colors.INFO}Using default configuration{Colors.END}")
    else:
        print(f"{Colors.YELLOW}{Emojis.WARNING} Config file not found: {config_file}{Colors.END}")
        print(f"{Colors.INFO}Using default configuration{Colors.END}")
    
    return default_config

def create_default_config():
    """Create default configuration file"""
    default_config = {
        "interface": "wg0",
        "peer_ip": "",
        "mtu_min": 1280,
        "mtu_max": 1500, 
        "mtu_step": 20,
        "test_duration": 10,
        "mtu_change_delay": 2,
        "messenger_port": 12345,
        "log_level": "INFO",
        "log_dir": "./logs",
        "output_dir": "./results",
        "binary_search": False,
        "generate_heatmap": True,
        "randomize_order": False
    }
    
    with open('mtu_finder_config.json', 'w') as f:
        json.dump(default_config, f, indent=2)
        
    print(f"{Colors.GREEN}{Emojis.CHECK} Created default config: mtu_finder_config.json{Colors.END}")

def main():
    parser = argparse.ArgumentParser(description='WireGuard MTU Finder')
    parser.add_argument('mode', choices=['server', 'peer'], 
                       help='Run mode: server (coordinator) or peer (responder)')
    parser.add_argument('--config', '-c', default='mtu_finder_config.json',
                       help='Configuration file path')
    parser.add_argument('--peer-ip', help='Peer IP address (server mode)')
    parser.add_argument('--interface', '-i', default='wg0', 
                       help='WireGuard interface name')
    parser.add_argument('--binary-search', '-b', action='store_true',
                       help='Use binary search instead of full matrix')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config()
        return
        
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.peer_ip:
        config['peer_ip'] = args.peer_ip
    if args.interface:
        config['interface'] = args.interface
    if args.binary_search:
        config['binary_search'] = True
        
    # Validate required settings
    if args.mode == 'server' and not config.get('peer_ip'):
        print(f"{Colors.RED}{Emojis.CROSS} Peer IP is required for server mode{Colors.END}")
        print(f"{Colors.YELLOW}{Emojis.INFO} Use --peer-ip option or set 'peer_ip' in config file{Colors.END}")
        sys.exit(1)
    
    # Validate interface exists if specified
    interface = config.get('interface', 'wg0')
    try:
        result = subprocess.run(['ip', 'link', 'show', interface], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{Colors.YELLOW}{Emojis.WARNING} Interface {interface} not found{Colors.END}")
            print(f"{Colors.INFO}Available interfaces:{Colors.END}")
            subprocess.run(['ip', 'link', 'show'])
    except Exception:
        pass  # Continue anyway, let the script handle interface issues
        
    # Create and run MTU finder
    finder = WireGuardMTUFinder(config)
    
    try:
        if args.mode == 'server':
            finder.run_server_mode()
        else:
            finder.run_peer_mode()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}{Emojis.WARNING} Operation interrupted by user{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}{Emojis.CROSS} Error: {e}{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()
