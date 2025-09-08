#!/usr/bin/env python3
"""
Progressive Frequency Test

Tests read-read-write pattern (leader read → follower read → follower write) 
with increasing follower motor count and frequency to find limits.
"""

import json
import time
import traceback
from pathlib import Path
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration


def load_calibration(cal_path):
    """Load calibration from JSON file"""
    with open(cal_path) as f:
        cal_data = json.load(f)
    calibration = {}
    for motor_name, cal_vals in cal_data.items():
        calibration[motor_name] = MotorCalibration(
            id=cal_vals['id'],
            drive_mode=cal_vals['drive_mode'], 
            homing_offset=cal_vals['homing_offset'],
            range_min=cal_vals['range_min'],
            range_max=cal_vals['range_max']
        )
    return calibration


class ProgressiveFrequencyTest:
    def __init__(self, leader_port, leader_cal_path, follower_port, follower_cal_path):
        # Leader always has all motors
        self.leader_config = SO101LeaderConfig(
            id='dum_e_leader',
            port=leader_port,
            use_degrees=False,
            calibration_dir=Path(leader_cal_path).parent
        )
        
        self.follower_port = follower_port
        self.follower_cal_path = follower_cal_path
        self.follower_calibration = load_calibration(follower_cal_path)
        
        # Motor definitions
        norm_mode_body = MotorNormMode.RANGE_M100_100
        self.all_motors = {
            'shoulder_pan': Motor(1, 'sts3215', norm_mode_body),
            'shoulder_lift': Motor(2, 'sts3215', norm_mode_body), 
            'elbow_flex': Motor(3, 'sts3215', norm_mode_body),
            'wrist_flex': Motor(4, 'sts3215', norm_mode_body),
            'wrist_roll': Motor(5, 'sts3215', norm_mode_body),
            'gripper': Motor(6, 'sts3215', MotorNormMode.RANGE_0_100),
        }
        
        self.motor_names = list(self.all_motors.keys())
        self.leader = None

    def connect_leader(self):
        """Connect leader with all motors"""
        try:
            self.leader = SO101Leader(self.leader_config)
            self.leader.connect(calibrate=False)
            print("✅ Leader connected with all 6 motors")
            return True
        except Exception as e:
            print(f"❌ Leader connection failed: {e}")
            return False

    def disconnect_leader(self):
        """Disconnect leader"""
        if self.leader:
            try:
                self.leader.disconnect()
                print("✅ Leader disconnected")
            except Exception as e:
                print(f"⚠️ Leader disconnect error: {e}")

    def create_follower_subset(self, num_motors):
        """Create follower configuration with specified number of motors"""
        subset_motors = dict(list(self.all_motors.items())[:num_motors])
        subset_calibration = dict(list(self.follower_calibration.items())[:num_motors])
        
        # Create follower config with subset
        follower_config = SO101FollowerConfig(
            id='dum_e_follower',
            port=self.follower_port,
            use_degrees=False,
            cameras={},
            calibration_dir=Path(self.follower_cal_path).parent,
            max_relative_target=None
        )
        
        return follower_config, subset_motors, subset_calibration

    def test_frequency_for_motor_count(self, num_motors, test_frequencies, test_duration=3.0):
        """Test different frequencies for a specific number of follower motors"""
        print(f"\n--- Testing with {num_motors} follower motors ---")
        motor_ids = list(range(1, num_motors + 1))
        print(f"    Follower motor IDs: {motor_ids}")
        
        frequency_results = {}
        
        for freq in test_frequencies:
            print(f"  Testing {freq}Hz...", end=" ", flush=True)
            
            try:
                # Create follower with subset of motors
                follower_config, subset_motors, subset_calibration = self.create_follower_subset(num_motors)
                
                # Use direct bus creation for more control
                follower_bus = FeetechMotorsBus(self.follower_port, subset_motors, calibration=subset_calibration)
                follower_bus.connect()
                
                # Test the read-read-write pattern at this frequency
                period = 1.0 / freq
                start_time = time.time()
                success_count = 0
                total_attempts = 0
                failure_messages = []
                
                while time.time() - start_time < test_duration:
                    loop_start = time.time()
                    
                    try:
                        # Pattern: leader read → follower read → follower write
                        
                        # 1. Leader read (all 6 motors)
                        leader_action = self.leader.get_action()
                        
                        # 2. Follower read (subset of motors)
                        follower_positions = follower_bus.sync_read('Present_Position', num_retry=3)
                        
                        # 3. Follower write (subset of motors) - use follower observation for corresponding motors
                        # basically don't move the follower.
                        write_data = {}
                        for motor_name in subset_motors.keys():
                            if f"{motor_name}.pos" in leader_action:
                                write_data[motor_name] = follower_positions[f"{motor_name}.pos"]
                        
                        if write_data:
                            follower_bus.sync_write('Goal_Position', write_data, num_retry=3)
                        
                        success_count += 1
                        
                    except Exception as e:
                        if len(failure_messages) < 3:  # Store first few error messages
                            failure_messages.append(str(e)[:60] + "...")
                    
                    total_attempts += 1
                    
                    # Maintain frequency
                    elapsed = time.time() - loop_start
                    if elapsed < period:
                        time.sleep(period - elapsed)
                
                # Calculate results
                success_rate = (success_count / total_attempts) * 100 if total_attempts > 0 else 0
                actual_freq = total_attempts / test_duration
                
                frequency_results[freq] = {
                    'success_rate': success_rate,
                    'actual_freq': actual_freq,
                    'total_attempts': total_attempts,
                    'success_count': success_count,
                    'failure_messages': failure_messages
                }
                
                # Disconnect follower bus
                follower_bus.disconnect()
                
                # Report result
                status = "✅" if success_rate >= 90 else "⚠️" if success_rate >= 70 else "❌"
                print(f"{status} {success_rate:.1f}% ({success_count}/{total_attempts})")
                
                # If frequency fails badly, don't test higher frequencies
                if success_rate < 50:
                    print(f"    🛑 Stopping at {freq}Hz due to poor reliability")
                    break
                    
            except Exception as e:
                print(f"❌ FAILED - {e}")
                frequency_results[freq] = {
                    'success_rate': 0,
                    'error': str(e)
                }
                break
            
            time.sleep(0.5)  # Brief recovery between frequencies
        
        return frequency_results

    def find_max_frequency_for_motor_count(self, num_motors):
        """Find maximum sustainable frequency for a given number of motors"""
        # Start with reasonable frequency range
        test_frequencies = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        
        results = self.test_frequency_for_motor_count(num_motors, test_frequencies)
        
        # Find max frequency with >90% success rate
        max_freq = 0
        best_result = None
        
        for freq, result in results.items():
            if result.get('success_rate', 0) >= 90:
                if freq > max_freq:
                    max_freq = freq
                    best_result = result
        
        return max_freq, best_result, results

    def run_progressive_test(self):
        """Run the full progressive test"""
        print("Progressive Motor Count vs Frequency Test")
        print("=" * 60)
        print("Pattern: Leader Read → Follower Read → Follower Write")
        print("Leader: Always 6 motors | Follower: Progressive 1-6 motors")
        print("=" * 60)
        
        if not self.connect_leader():
            return None
        
        try:
            overall_results = {}
            
            # Test with 1 to 6 follower motors
            for num_motors in range(1, 7):
                max_freq, best_result, all_results = self.find_max_frequency_for_motor_count(num_motors)
                
                overall_results[num_motors] = {
                    'max_frequency': max_freq,
                    'best_result': best_result,
                    'all_results': all_results
                }
                
                print(f"\n  📊 RESULT: {num_motors} follower motors → Max {max_freq}Hz")
                if best_result:
                    print(f"      Success: {best_result['success_count']}/{best_result['total_attempts']} operations")
                    print(f"      Actual freq: {best_result['actual_freq']:.1f}Hz")
                
                time.sleep(1.0)  # Recovery between motor count tests
            
            # Summary table
            print(f"\n{'='*60}")
            print("PROGRESSIVE TEST SUMMARY")
            print("=" * 60)
            print(f"{'Motors':<8} | {'Max Hz':<8} | {'Success Rate':<12} | {'Status'}")
            print("-" * 60)
            
            for num_motors, result in overall_results.items():
                max_freq = result['max_frequency']
                if result['best_result']:
                    success_rate = result['best_result']['success_rate']
                    status = "✅ Good" if max_freq >= 30 else "⚠️ Limited" if max_freq >= 20 else "❌ Poor"
                else:
                    success_rate = 0
                    status = "❌ Failed"
                
                print(f"{num_motors:<8} | {max_freq:<8} | {success_rate:<11.1f}% | {status}")
            
            # Analysis and recommendations
            print(f"\n{'='*60}")
            print("ANALYSIS & RECOMMENDATIONS")
            print("=" * 60)
            
            # Find the sweet spot
            motor_counts_30hz = [n for n, r in overall_results.items() if r['max_frequency'] >= 30]
            
            if motor_counts_30hz:
                max_motors_30hz = max(motor_counts_30hz)
                print(f"✅ 30Hz ACHIEVABLE with up to {max_motors_30hz} follower motors")
                print(f"   Recommendation: Use {max_motors_30hz} motors for 30Hz camera sync")
                
                if max_motors_30hz < 6:
                    unused_motors = 6 - max_motors_30hz
                    unused_names = self.motor_names[max_motors_30hz:]
                    print(f"   📝 Disable these {unused_motors} motors: {unused_names}")
                    
            else:
                best_compromise = max(overall_results.items(), key=lambda x: x[1]['max_frequency'])
                motor_count, result = best_compromise
                print(f"❌ 30Hz NOT achievable with current setup")
                print(f"   Best compromise: {motor_count} motors at {result['max_frequency']}Hz")
                print(f"   Consider: Reduce motor count or accept lower frequency")
            
            # Pattern efficiency analysis
            degradation_pattern = []
            for i in range(1, 7):
                freq = overall_results[i]['max_frequency']
                degradation_pattern.append((i, freq))
            
            print(f"\n📈 DEGRADATION PATTERN:")
            for motors, freq in degradation_pattern:
                if motors == 1:
                    baseline = freq
                    print(f"   {motors} motor:  {freq}Hz (baseline)")
                else:
                    loss = baseline - freq if baseline > 0 else 0
                    print(f"   {motors} motors: {freq}Hz (-{loss}Hz from baseline)")
            
            return overall_results
            
        except Exception as e:
            print(f"Test failed: {e}")
            traceback.print_exc()
            return None
        finally:
            self.disconnect_leader()


def main():
    """Main test function"""
    
    # Device configuration
    leader_port = '/dev/so101_leader'
    leader_cal = '/home/melon/sherry/lerobot/calibration/teleoperators/so101_leader/dum_e_leader.json'
    follower_port = '/dev/so101_follower'
    follower_cal = '/home/melon/sherry/lerobot/calibration/robots/so101_follower/dum_e_follower.json'
    
    test = ProgressiveFrequencyTest(leader_port, leader_cal, follower_port, follower_cal)
    results = test.run_progressive_test()
    
    if results:
        print(f"\n🎯 KEY FINDING:")
        motor_30hz = [n for n, r in results.items() if r['max_frequency'] >= 30]
        if motor_30hz:
            print(f"   Use {max(motor_30hz)} follower motors to achieve your 30Hz requirement")
        else:
            best = max(results.items(), key=lambda x: x[1]['max_frequency'])
            print(f"   Maximum achievable: {best[1]['max_frequency']}Hz with {best[0]} motors")


if __name__ == '__main__':
    main()