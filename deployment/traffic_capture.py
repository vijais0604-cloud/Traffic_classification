# capture.py

import subprocess

def capture():
    subprocess.run([
        "sudo",
        "tcpdump",
        "-i",
        "en0",
        "-G",
        "30",
        "-W",
        "1",
        "-w",
        "traffic.pcap"
    ])