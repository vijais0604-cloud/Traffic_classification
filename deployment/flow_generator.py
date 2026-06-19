# flow_generator.py

import subprocess

def generate_csv():

    subprocess.run([
        "uv",
        "run",
        "cicflowmeter",
        "-f",
        "traffic.pcap",
        "-c",
        "flows.csv"
    ])