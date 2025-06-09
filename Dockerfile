FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    iperf3 \
    iproute2 \
    iputils-ping \
    sudo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY wg_mtu_finder.py .
COPY config.json .

RUN chmod +x wg_mtu_finder.py

CMD ["python3", "wg_mtu_finder.py", "--help"]