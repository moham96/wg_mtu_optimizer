#!/bin/bash
set -e

echo "Installing WireGuard MTU Finder..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "Please do not run this installer as root"
   echo "The script itself needs root privileges, but the installer should run as regular user"
   exit 1
fi

# Update package list
echo "Updating package list..."
sudo apt update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y python3 python3-pip iperf3 iproute2 iputils-ping

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --user -r requirements.txt

# Make script executable
chmod +x wg_mtu_finder.py

# Create symlink for easy access (optional)
if [ ! -f /usr/local/bin/wg-mtu-finder ]; then
    sudo ln -s "$(pwd)/wg_mtu_finder.py" /usr/local/bin/wg-mtu-finder
    echo "Created symlink: /usr/local/bin/wg-mtu-finder"
fi

echo "Installation complete!"
echo ""
echo "Usage:"
echo "  Server: sudo wg-mtu-finder --mode server"
echo "  Client: sudo wg-mtu-finder --mode client --server-ip <SERVER_IP>"
echo ""
echo "For detailed instructions, see README.md"
