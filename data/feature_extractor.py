"""
Feature extraction for network traffic captured from hping3 attacks.
This module handles PCAP processing and converts traffic to features
compatible with our DoS detection models.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import subprocess
import tempfile
from typing import Dict, List, Optional, Union, Tuple
import pickle
import datetime
import json

# Try to import specialized libraries, with fallbacks
try:
    import pyshark
    PYSHARK_AVAILABLE = True
except ImportError:
    PYSHARK_AVAILABLE = False
    logging.warning("pyshark not available - limited PCAP processing capabilities")

try:
    from scapy.all import rdpcap, IP, TCP, UDP
    SCAPY_AVAILABLE = True 
except ImportError:
    SCAPY_AVAILABLE = False
    logging.warning("scapy not available - limited PCAP processing capabilities")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define flow timeout for session aggregation (in seconds)
FLOW_TIMEOUT = 120  # 2 minutes

def check_cicflowmeter_available():
    """
    Check if CICFlowMeter is available in the system.
    
    Returns:
        Boolean indicating if CICFlowMeter is available
    """
    try:
        result = subprocess.run(
            ["cicflowmeter", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

def extract_features_cicflowmeter(pcap_file: str, output_file: str) -> str:
    """
    Extract features using CICFlowMeter.
    
    Args:
        pcap_file: Path to the PCAP file
        output_file: Path for the output CSV file
        
    Returns:
        Path to the generated CSV file
    """
    logger.info(f"Extracting features from {pcap_file} using CICFlowMeter")
    
    try:
        cmd = ["cicflowmeter", "-f", pcap_file, "-c", output_file]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Error running CICFlowMeter: {result.stderr}")
            raise RuntimeError(f"CICFlowMeter failed: {result.stderr}")
        
        logger.info(f"Successfully extracted features to {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error using CICFlowMeter: {e}")
        raise

def extract_features_pyshark(pcap_file: str, output_file: str) -> str:
    """
    Extract features using pyshark when CICFlowMeter is not available.
    
    Args:
        pcap_file: Path to the PCAP file
        output_file: Path for the output CSV file
        
    Returns:
        Path to the generated CSV file
    """
    if not PYSHARK_AVAILABLE:
        raise ImportError("pyshark not installed - cannot process PCAP file")
    
    logger.info(f"Extracting features from {pcap_file} using pyshark")
    
    try:
        # Open the pcap file
        cap = pyshark.FileCapture(pcap_file)
        
        # Track flows using a dictionary
        flows = {}
        
        # Process each packet
        packet_count = 0
        for packet in cap:
            packet_count += 1
            if packet_count % 1000 == 0:
                logger.info(f"Processed {packet_count} packets")
            
            try:
                if not hasattr(packet, 'ip'):
                    continue
                
                # Define flow key based on IP and port information
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                
                # Get protocol and port information if available
                if hasattr(packet, 'tcp'):
                    protocol = 'tcp'
                    src_port = int(packet.tcp.srcport)
                    dst_port = int(packet.tcp.dstport)
                    flags = {
                        'syn': int(hasattr(packet.tcp, 'flags_syn') and packet.tcp.flags_syn == '1'),
                        'ack': int(hasattr(packet.tcp, 'flags_ack') and packet.tcp.flags_ack == '1'),
                        'fin': int(hasattr(packet.tcp, 'flags_fin') and packet.tcp.flags_fin == '1'),
                        'rst': int(hasattr(packet.tcp, 'flags_reset') and packet.tcp.flags_reset == '1'),
                        'psh': int(hasattr(packet.tcp, 'flags_push') and packet.tcp.flags_push == '1')
                    }
                elif hasattr(packet, 'udp'):
                    protocol = 'udp'
                    src_port = int(packet.udp.srcport)
                    dst_port = int(packet.udp.dstport)
                    flags = {'syn': 0, 'ack': 0, 'fin': 0, 'rst': 0, 'psh': 0}
                else:
                    continue  # Skip non-TCP/UDP packets
                
                # Create bidirectional flow key
                forward_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
                backward_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
                
                # Determine if packet belongs to an existing flow
                if forward_key in flows:
                    flow_key = forward_key
                    direction = 'forward'
                elif backward_key in flows:
                    flow_key = backward_key
                    direction = 'backward'
                else:
                    # Create new flow
                    flow_key = forward_key
                    direction = 'forward'
                    flows[flow_key] = {
                        'src_ip': src_ip,
                        'dst_ip': dst_ip,
                        'src_port': src_port,
                        'dst_port': dst_port,
                        'protocol': protocol,
                        'start_time': float(packet.sniff_timestamp),
                        'last_time': float(packet.sniff_timestamp),
                        'fwd_packets': 0,
                        'bwd_packets': 0,
                        'fwd_bytes': 0,
                        'bwd_bytes': 0,
                        'packet_times': [],
                        'packet_sizes': [],
                        'fwd_packet_sizes': [],
                        'bwd_packet_sizes': [],
                        'syn_count': 0,
                        'ack_count': 0,
                        'fin_count': 0,
                        'rst_count': 0,
                        'psh_count': 0
                    }
                
                # Get packet length
                pkt_len = int(packet.length)
                
                # Update flow statistics
                flow = flows[flow_key]
                flow['last_time'] = float(packet.sniff_timestamp)
                flow['packet_times'].append(float(packet.sniff_timestamp))
                flow['packet_sizes'].append(pkt_len)
                
                # Update direction-specific statistics
                if direction == 'forward':
                    flow['fwd_packets'] += 1
                    flow['fwd_bytes'] += pkt_len
                    flow['fwd_packet_sizes'].append(pkt_len)
                else:
                    flow['bwd_packets'] += 1
                    flow['bwd_bytes'] += pkt_len
                    flow['bwd_packet_sizes'].append(pkt_len)
                
                # Update flag counts
                flow['syn_count'] += flags['syn']
                flow['ack_count'] += flags['ack']
                flow['fin_count'] += flags['fin']
                flow['rst_count'] += flags['rst']
                flow['psh_count'] += flags['psh']
            
            except Exception as e:
                logger.warning(f"Error processing packet: {e}")
                continue
        
        cap.close()
        
        # Convert flows to CICFlowMeter-like features
        flow_features = []
        
        for flow_id, flow in flows.items():
            # Skip flows with too few packets
            if flow['fwd_packets'] + flow['bwd_packets'] < 2:
                continue
                
            # Calculate basic flow features
            duration = max(0.001, flow['last_time'] - flow['start_time'])
            flow_bytes = flow['fwd_bytes'] + flow['bwd_bytes']
            flow_pkts = flow['fwd_packets'] + flow['bwd_packets']
            
            # Calculate packet rate and byte rate
            pkt_rate = flow_pkts / duration
            byte_rate = flow_bytes / duration
            
            # Calculate inter-arrival times if possible
            iat_mean = 0
            iat_std = 0
            iat_max = 0
            iat_min = 0
            
            if len(flow['packet_times']) > 1:
                sorted_times = sorted(flow['packet_times'])
                iats = [sorted_times[i] - sorted_times[i-1] for i in range(1, len(sorted_times))]
                if iats:
                    iat_mean = np.mean(iats)
                    iat_std = np.std(iats)
                    iat_max = np.max(iats)
                    iat_min = np.min(iats)
            
            # Calculate packet size statistics
            pkt_size_mean = np.mean(flow['packet_sizes'])
            pkt_size_std = np.std(flow['packet_sizes']) if len(flow['packet_sizes']) > 1 else 0
            
            # Prepare feature dictionary
            feature_dict = {
                'Flow ID': flow_id,
                'Source IP': flow['src_ip'],
                'Source Port': flow['src_port'],
                'Destination IP': flow['dst_ip'],
                'Destination Port': flow['dst_port'],
                'Protocol': 6 if flow['protocol'] == 'tcp' else 17,  # 6=TCP, 17=UDP
                'Timestamp': datetime.datetime.fromtimestamp(flow['start_time']).strftime('%Y-%m-%d %H:%M:%S'),
                'Flow Duration': duration * 1000,  # Convert to milliseconds
                'Total Fwd Packets': flow['fwd_packets'],
                'Total Backward Packets': flow['bwd_packets'],
                'Total Length of Fwd Packets': flow['fwd_bytes'],
                'Total Length of Bwd Packets': flow['bwd_bytes'],
                'Fwd Packet Length Max': np.max(flow['fwd_packet_sizes']) if flow['fwd_packet_sizes'] else 0,
                'Fwd Packet Length Min': np.min(flow['fwd_packet_sizes']) if flow['fwd_packet_sizes'] else 0,
                'Fwd Packet Length Mean': np.mean(flow['fwd_packet_sizes']) if flow['fwd_packet_sizes'] else 0,
                'Fwd Packet Length Std': np.std(flow['fwd_packet_sizes']) if len(flow['fwd_packet_sizes']) > 1 else 0,
                'Bwd Packet Length Max': np.max(flow['bwd_packet_sizes']) if flow['bwd_packet_sizes'] else 0,
                'Bwd Packet Length Min': np.min(flow['bwd_packet_sizes']) if flow['bwd_packet_sizes'] else 0,
                'Bwd Packet Length Mean': np.mean(flow['bwd_packet_sizes']) if flow['bwd_packet_sizes'] else 0,
                'Bwd Packet Length Std': np.std(flow['bwd_packet_sizes']) if len(flow['bwd_packet_sizes']) > 1 else 0,
                'Flow Bytes/s': byte_rate,
                'Flow Packets/s': pkt_rate,
                'Flow IAT Mean': iat_mean,
                'Flow IAT Std': iat_std,
                'Flow IAT Max': iat_max,
                'Flow IAT Min': iat_min,
                'Min Packet Length': min(flow['packet_sizes']) if flow['packet_sizes'] else 0,
                'Max Packet Length': max(flow['packet_sizes']) if flow['packet_sizes'] else 0,
                'Packet Length Mean': pkt_size_mean,
                'Packet Length Std': pkt_size_std,
                'Packet Length Variance': pkt_size_std**2,
                'FIN Flag Count': flow['fin_count'],
                'SYN Flag Count': flow['syn_count'],
                'RST Flag Count': flow['rst_count'],
                'PSH Flag Count': flow['psh_count'],
                'ACK Flag Count': flow['ack_count'],
                'URG Flag Count': 0,  # Not tracked
                'CWE Flag Count': 0,  # Not tracked
                'ECE Flag Count': 0,  # Not tracked
                'Down/Up Ratio': flow['bwd_bytes'] / max(1, flow['fwd_bytes']),
                'Average Packet Size': pkt_size_mean,
                'Avg Fwd Segment Size': np.mean(flow['fwd_packet_sizes']) if flow['fwd_packet_sizes'] else 0,
                'Avg Bwd Segment Size': np.mean(flow['bwd_packet_sizes']) if flow['bwd_packet_sizes'] else 0,
                # We'll set reasonable defaults for any other required features
            }
            
            flow_features.append(feature_dict)
        
        # Create DataFrame
        df = pd.DataFrame(flow_features)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        logger.info(f"Extracted {len(flow_features)} flows and saved to {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error extracting features with pyshark: {e}")
        raise

def extract_features_scapy(pcap_file: str, output_file: str) -> str:
    """
    Extract features using scapy when CICFlowMeter and pyshark are not available.
    
    Args:
        pcap_file: Path to the PCAP file
        output_file: Path for the output CSV file
        
    Returns:
        Path to the generated CSV file
    """
    if not SCAPY_AVAILABLE:
        raise ImportError("scapy not installed - cannot process PCAP file")
    
    logger.info(f"Extracting features from {pcap_file} using scapy")
    
    try:
        # Read packets from pcap file
        packets = rdpcap(pcap_file)
        logger.info(f"Loaded {len(packets)} packets from {pcap_file}")
        
        # Track flows using a dictionary
        flows = {}
        
        # Process each packet
        for i, packet in enumerate(packets):
            if i % 1000 == 0:
                logger.info(f"Processed {i} packets")
            
            try:
                if IP not in packet:
                    continue
                
                # Extract IP layer information
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                
                # Handle TCP packets
                if TCP in packet:
                    protocol = 'tcp'
                    src_port = packet[TCP].sport
                    dst_port = packet[TCP].dport
                    flags = {
                        'syn': 1 if packet[TCP].flags & 0x02 else 0,  # SYN flag
                        'ack': 1 if packet[TCP].flags & 0x10 else 0,  # ACK flag
                        'fin': 1 if packet[TCP].flags & 0x01 else 0,  # FIN flag
                        'rst': 1 if packet[TCP].flags & 0x04 else 0,  # RST flag
                        'psh': 1 if packet[TCP].flags & 0x08 else 0,  # PSH flag
                    }
                # Handle UDP packets
                elif UDP in packet:
                    protocol = 'udp'
                    src_port = packet[UDP].sport
                    dst_port = packet[UDP].dport
                    flags = {'syn': 0, 'ack': 0, 'fin': 0, 'rst': 0, 'psh': 0}
                else:
                    continue  # Skip non-TCP/UDP packets
                
                # Create bidirectional flow key
                forward_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
                backward_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
                
                # Get packet timestamp
                timestamp = float(packet.time)
                
                # Determine if packet belongs to an existing flow
                if forward_key in flows:
                    flow_key = forward_key
                    direction = 'forward'
                elif backward_key in flows:
                    flow_key = backward_key
                    direction = 'backward'
                else:
                    # Create new flow
                    flow_key = forward_key
                    direction = 'forward'
                    flows[flow_key] = {
                        'src_ip': src_ip,
                        'dst_ip': dst_ip,
                        'src_port': src_port,
                        'dst_port': dst_port,
                        'protocol': protocol,
                        'start_time': timestamp,
                        'last_time': timestamp,
                        'fwd_packets': 0,
                        'bwd_packets': 0,
                        'fwd_bytes': 0,
                        'bwd_bytes': 0,
                        'packet_times': [],
                        'packet_sizes': [],
                        'fwd_packet_sizes': [],
                        'bwd_packet_sizes': [],
                        'syn_count': 0,
                        'ack_count': 0,
                        'fin_count': 0,
                        'rst_count': 0,
                        'psh_count': 0
                    }
                
                # Get packet length
                pkt_len = len(packet)
                
                # Update flow statistics
                flow = flows[flow_key]
                flow['last_time'] = timestamp
                flow['packet_times'].append(timestamp)
                flow['packet_sizes'].append(pkt_len)
                
                # Update direction-specific statistics
                if direction == 'forward':
                    flow['fwd_packets'] += 1
                    flow['fwd_bytes'] += pkt_len
                    flow['fwd_packet_sizes'].append(pkt_len)
                else:
                    flow['bwd_packets'] += 1
                    flow['bwd_bytes'] += pkt_len
                    flow['bwd_packet_sizes'].append(pkt_len)
                
                # Update flag counts
                flow['syn_count'] += flags['syn']
                flow['ack_count'] += flags['ack']
                flow['fin_count'] += flags['fin']
                flow['rst_count'] += flags['rst']
                flow['psh_count'] += flags['psh']
            
            except Exception as e:
                logger.warning(f"Error processing packet {i}: {e}")
                continue
        
        # Convert flows to CICFlowMeter-like features
        flow_features = []
        
        for flow_id, flow in flows.items():
            # Skip flows with too few packets
            if flow['fwd_packets'] + flow['bwd_packets'] < 2:
                continue
                
            # Calculate basic flow features
            duration = max(0.001, flow['last_time'] - flow['start_time'])
            flow_bytes = flow['fwd_bytes'] + flow['bwd_bytes']
            flow_pkts = flow['fwd_packets'] + flow['bwd_packets']
            
            # Calculate packet rate and byte rate
            pkt_rate = flow_pkts / duration
            byte_rate = flow_bytes / duration
            
            # Calculate inter-arrival times if possible
            iat_mean = 0
            iat_std = 0
            iat_max = 0
            iat_min = 0
            
            if len(flow['packet_times']) > 1:
                sorted_times = sorted(flow['packet_times'])
                iats = [sorted_times[i] - sorted_times[i-1] for i in range(1, len(sorted_times))]
                if iats:
                    iat_mean = np.mean(iats)
                    iat_std = np.std(iats)
                    iat_max = np.max(iats)
                    iat_min = np.min(iats)
            
            # Calculate packet size statistics
            pkt_size_mean = np.mean(flow['packet_sizes'])
            pkt_size_std = np.std(flow['packet_sizes']) if len(flow['packet_sizes']) > 1 else 0
            
            # Prepare feature dictionary
            feature_dict = {
                'Flow ID': flow_id,
                'Source IP': flow['src_ip'],
                'Source Port': flow['src_port'],
                'Destination IP': flow['dst_ip'],
                'Destination Port': flow['dst_port'],
                'Protocol': 6 if flow['protocol'] == 'tcp' else 17,  # 6=TCP, 17=UDP
                'Timestamp': datetime.datetime.fromtimestamp(flow['start_time']).strftime('%Y-%m-%d %H:%M:%S'),
                'Flow Duration': duration * 1000,  # Convert to milliseconds
                'Total Fwd Packets': flow['fwd_packets'],
                'Total Backward Packets': flow['bwd_packets'],
                'Total Length of Fwd Packets': flow['fwd_bytes'],
                'Total Length of Bwd Packets': flow['bwd_bytes'],
                'Fwd Packet Length Max': np.max(flow['fwd_packet_sizes']) if flow['fwd_packet_sizes'] else 0,
                'Fwd Packet Length Min': np.min(flow['fwd_packet_sizes']) if flow['fwd_packet_sizes'] else 0,
                'Fwd Packet Length Mean': np.mean(flow['fwd_packet_sizes']) if flow['fwd_packet_sizes'] else 0,
                'Fwd Packet Length Std': np.std(flow['fwd_packet_sizes']) if len(flow['fwd_packet_sizes']) > 1 else 0,
                'Bwd Packet Length Max': np.max(flow['bwd_packet_sizes']) if flow['bwd_packet_sizes'] else 0,
                'Bwd Packet Length Min': np.min(flow['bwd_packet_sizes']) if flow['bwd_packet_sizes'] else 0,
                'Bwd Packet Length Mean': np.mean(flow['bwd_packet_sizes']) if flow['bwd_packet_sizes'] else 0,
                'Bwd Packet Length Std': np.std(flow['bwd_packet_sizes']) if len(flow['bwd_packet_sizes']) > 1 else 0,
                'Flow Bytes/s': byte_rate,
                'Flow Packets/s': pkt_rate,
                'Flow IAT Mean': iat_mean,
                'Flow IAT Std': iat_std,
                'Flow IAT Max': iat_max,
                'Flow IAT Min': iat_min,
                'Min Packet Length': min(flow['packet_sizes']) if flow['packet_sizes'] else 0,
                'Max Packet Length': max(flow['packet_sizes']) if flow['packet_sizes'] else 0,
                'Packet Length Mean': pkt_size_mean,
                'Packet Length Std': pkt_size_std,
                'Packet Length Variance': pkt_size_std**2,
                'FIN Flag Count': flow['fin_count'],
                'SYN Flag Count': flow['syn_count'],
                'RST Flag Count': flow['rst_count'],
                'PSH Flag Count': flow['psh_count'],
                'ACK Flag Count': flow['ack_count'],
                'URG Flag Count': 0,  # Not tracked
                'CWE Flag Count': 0,  # Not tracked
                'ECE Flag Count': 0,  # Not tracked
                'Down/Up Ratio': flow['bwd_bytes'] / max(1, flow['fwd_bytes']),
                'Average Packet Size': pkt_size_mean,
                'Avg Fwd Segment Size': np.mean(flow['fwd_packet_sizes']) if flow['fwd_packet_sizes'] else 0,
                'Avg Bwd Segment Size': np.mean(flow['bwd_packet_sizes']) if flow['bwd_packet_sizes'] else 0,
                # Set reasonable defaults for any remaining features
            }
            
            flow_features.append(feature_dict)
        
        # Create DataFrame
        df = pd.DataFrame(flow_features)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        logger.info(f"Extracted {len(flow_features)} flows and saved to {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error extracting features with scapy: {e}")
        raise

def process_pcap_file(pcap_file: str, output_csv: str = None) -> str:
    """
    Process a PCAP file to extract features for DoS detection.
    
    Args:
        pcap_file: Path to the PCAP file
        output_csv: Optional path for the output CSV file
        
    Returns:
        Path to the generated CSV file
    """
    if not os.path.exists(pcap_file):
        raise FileNotFoundError(f"PCAP file not found: {pcap_file}")
    
    # Create output path if not provided
    if output_csv is None:
        output_csv = os.path.splitext(pcap_file)[0] + "_features.csv"
    
    logger.info(f"Processing PCAP file: {pcap_file}")
    logger.info(f"Output will be saved to: {output_csv}")
    
    # Try different feature extraction methods in order of preference
    if check_cicflowmeter_available():
        logger.info("Using CICFlowMeter for feature extraction")
        return extract_features_cicflowmeter(pcap_file, output_csv)
    elif PYSHARK_AVAILABLE:
        logger.info("Using pyshark for feature extraction")
        return extract_features_pyshark(pcap_file, output_csv)
    elif SCAPY_AVAILABLE:
        logger.info("Using scapy for feature extraction")
        return extract_features_scapy(pcap_file, output_csv)
    else:
        raise RuntimeError("No suitable feature extraction method available. Please install CICFlowMeter, pyshark, or scapy.")

def capture_live_traffic(interface: str, output_pcap: str, duration: int = 60, bpf_filter: str = None) -> str:
    """
    Capture live network traffic to a PCAP file.
    
    Args:
        interface: Network interface to capture from
        output_pcap: Path to save the PCAP file
        duration: Duration of capture in seconds
        bpf_filter: Optional BPF filter for packet capture
        
    Returns:
        Path to the generated PCAP file
    """
    logger.info(f"Capturing live traffic from interface {interface} for {duration} seconds")
    
    # Use tcpdump for capture (works on most Unix-like systems)
    cmd = ["tcpdump", "-i", interface, "-w", output_pcap]
    
    # Add filter if provided
    if bpf_filter:
        cmd.extend(["-f", bpf_filter])
    
    try:
        # Start capture process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for specified duration
        logger.info(f"Capture started. Waiting for {duration} seconds...")
        time.sleep(duration)
        
        # Terminate capture
        process.terminate()
        process.wait()
        
        logger.info(f"Capture completed. PCAP file saved to {output_pcap}")
        return output_pcap
    
    except Exception as e:
        logger.error(f"Error during packet capture: {e}")
        raise

def capture_and_analyze_hping3_attack(
    interface: str,
    target_ip: str,
    output_dir: str,
    attack_type: str = "syn",
    duration: int = 30,
    port: int = 80
) -> Dict[str, str]:
    """
    Capture and analyze an hping3 attack.
    
    Args:
        interface: Network interface to capture from
        target_ip: Target IP address of the attack
        output_dir: Directory to save output files
        attack_type: Type of attack ("syn", "udp", "icmp")
        duration: Duration of capture in seconds
        port: Target port for the attack
        
    Returns:
        Dictionary with paths to output files
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define file paths
    pcap_file = os.path.join(output_dir, f"hping3_{attack_type}_{timestamp}.pcap")
    csv_file = os.path.join(output_dir, f"hping3_{attack_type}_{timestamp}_features.csv")
    
    # Create filter to capture attack traffic
    bpf_filter = f"host {target_ip}"
    
    # Start packet capture
    logger.info(f"Starting packet capture for {attack_type} attack to {target_ip}:{port}")
    pcap_path = capture_live_traffic(interface, pcap_file, duration, bpf_filter)
    
    # Process the PCAP file
    logger.info(f"Processing captured traffic for feature extraction")
    csv_path = process_pcap_file(pcap_path, csv_file)
    
    return {
        "pcap_file": pcap_path,
        "csv_file": csv_path,
        "attack_type": attack_type,
        "target_ip": target_ip,
        "target_port": port,
        "duration": duration
    }

def prepare_hping3_attack_script(
    target_ip: str,
    attack_type: str = "syn",
    port: int = 80,
    duration: int = 30,
    output_file: str = None
) -> str:
    """
    Generate a script to run an hping3 attack.
    
    Args:
        target_ip: Target IP address
        attack_type: Type of attack ("syn", "udp", "icmp")
        port: Target port 
        duration: Duration of attack in seconds
        output_file: Path to save the script
        
    Returns:
        Path to the generated script
    """
    if output_file is None:
        output_file = f"run_hping3_{attack_type}_attack.sh"
    
    logger.info(f"Generating hping3 attack script for {attack_type} attack to {target_ip}:{port}")
    
    # Create the script content
    script = "#!/bin/bash\n\n"
    script += f"# hping3 {attack_type.upper()} flood attack script\n"
    script += f"# Target: {target_ip}:{port}\n"
    script += f"# Duration: {duration} seconds\n\n"
    
    # Add command based on attack type
    if attack_type.lower() == "syn":
        script += f"sudo hping3 -S --flood -p {port} {target_ip} &\n"
    elif attack_type.lower() == "udp":
        script += f"sudo hping3 --udp --flood -p {port} {target_ip} &\n"
    elif attack_type.lower() == "icmp":
        script += f"sudo hping3 --icmp --flood {target_ip} &\n"
    elif attack_type.lower() == "fin":
        script += f"sudo hping3 -F --flood -p {port} {target_ip} &\n"
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")
    
    # Add timer to stop the attack
    script += f"\n# Sleep for the duration of the attack\n"
    script += f"sleep {duration}\n\n"
    script += f"# Kill the hping3 process\n"
    script += f"kill $(ps aux | grep hping3 | grep -v grep | awk '{{print $2}}')\n"
    
    # Write to file
    with open(output_file, "w") as f:
        f.write(script)
    
    # Make the script executable
    os.chmod(output_file, 0o755)
    
    logger.info(f"Attack script saved to {output_file}")
    return output_file