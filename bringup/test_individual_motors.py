#!/usr/bin/env python3
"""
Individual Motor Testing Script for SO-101 Arms

This script tests both leader and follower motor buses independently
to verify communication and calibration loading. It also tests the
SO101Leader and SO101Follower classes with their get_action() and
get_observation() methods.
"""

import json
import traceback
import time
from pathlib import Path
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
import logging
from lerobot.utils.utils import init_logging

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

def test_port_scan(port):
    """Test motor detection on a port"""
    print(f'Scanning port {port}...')
    try:
        scan_result = FeetechMotorsBus.scan_port(port)
        print(f'Scan result: {scan_result}')
        return scan_result
    except Exception as e:
        print(f'Scan failed: {e}')
        return None

def test_motor_bus(port, calibration_path, name):
    """Test motor bus with calibration"""
    print(f'\n=== Testing {name} port with calibration ===')
    try:
        # Motor configuration (same for both leader and follower)
        norm_mode_body = MotorNormMode.RANGE_M100_100
        motors = {
            'shoulder_pan': Motor(1, 'sts3215', norm_mode_body),
            'shoulder_lift': Motor(2, 'sts3215', norm_mode_body), 
            'elbow_flex': Motor(3, 'sts3215', norm_mode_body),
            'wrist_flex': Motor(4, 'sts3215', norm_mode_body),
            'wrist_roll': Motor(5, 'sts3215', norm_mode_body),
        }
        if name == 'LEADER':
            motors['gripper'] = Motor(6, 'sts3215', MotorNormMode.RANGE_0_100)
        
        # Load calibration and create bus
        calibration = load_calibration(calibration_path)
        bus = FeetechMotorsBus(port, motors, calibration=calibration)
        bus.connect()
        print(f'SUCCESS: {name} motor bus connected!')
        
        # Read current positions
        start_t = time.time()
        positions = bus.sync_read('Present_Position', num_retry=6)
        end_t = time.time()
        print(f'{name} time: {end_t - start_t}')
        print(f'{name} positions: {positions}')
        
        bus.disconnect()
        return True
        
    except Exception as e:
        print(f'{name} FAILED: {e}')
        return False

def test_so101_leader(port, calibration_path):
    """Test SO101Leader class with get_action() method"""
    print(f'\n=== Testing SO101Leader Class ===')
    try:
        # Create leader configuration
        leader_config = SO101LeaderConfig(
            id='dum_e_leader',
            port=port,
            use_degrees=False,
            calibration_dir=Path(calibration_path).parent
        )
        
        # Instantiate SO101Leader
        leader = SO101Leader(leader_config)
        print('SO101Leader instantiated successfully')
        
        # Connect to leader
        leader.connect(calibrate=False)  # Skip calibration for testing
        print('SO101Leader connected successfully')
        
        # Test get_action() method
        print('Testing leader.get_action()...')
        start_time = time.time()
        action = leader.get_action()
        end_time = time.time()
        
        print(f'Leader action read time: {(end_time - start_time)*1000:.1f}ms')
        print(f'Leader action: {action}')
        print(f'Action keys: {list(action.keys())}')
        print(f'Action types: {[(k, type(v)) for k, v in action.items()]}')
        
        # Disconnect
        leader.disconnect()
        print('SO101Leader disconnected successfully')
        return True
        
    except Exception as e:
        print(f'SO101Leader test FAILED: {e}')
        return False

def test_so101_follower(port, calibration_path):
    """Test SO101Follower class with get_observation() method"""
    print(f'\n=== Testing SO101Follower Class ===')
    try:
        # Create follower configuration
        follower_config = SO101FollowerConfig(
            id='dum_e_follower',
            port=port,
            use_degrees=False,
            cameras={},  # No cameras for this test
            calibration_dir=Path(calibration_path).parent
        )
        
        # Instantiate SO101Follower
        follower = SO101Follower(follower_config)
        print('SO101Follower instantiated successfully')
        
        # Connect to follower
        follower.connect(calibrate=False)  # Skip calibration for testing
        print('SO101Follower connected successfully')
        
        # Test get_observation() method
        print('Testing follower.get_observation()...')
        start_time = time.time()
        observation = follower.get_observation()
        end_time = time.time()
        
        print(f'Follower observation read time: {(end_time - start_time)*1000:.1f}ms')
        print(f'Observation keys: {list(observation.keys())}')
        print(f'Motor positions: {[(k, v) for k, v in observation.items() if k.endswith(".pos")]}')
        print(f'Timestamps: {[(k, v) for k, v in observation.items() if "timestamp" in k]}')
        
        # Disconnect
        follower.disconnect()
        print('SO101Follower disconnected successfully')
        return True
        
    except Exception as e:
        print(f'SO101Follower test FAILED: {e}')
        return False

def test_interleaved_10hz(leader_port, leader_cal_path, follower_port, follower_cal_path, duration=5.0):
    """Test interleaved get_action() and get_observation() at 10Hz"""
    target_freq = 30.0
    print(f'\n=== Testing Interleaved Operation at {target_freq}Hz ===')
    try:
        # Create configurations
        leader_config = SO101LeaderConfig(
            id='dum_e_leader',
            port=leader_port,
            use_degrees=False,
            calibration_dir=Path(leader_cal_path).parent
        )
        
        follower_config = SO101FollowerConfig(
            id='dum_e_follower',
            port=follower_port,
            use_degrees=False,
            camera_configs_path='/home/melon/sherry/so101_bench/bringup/camera_configs.json',
            cameras={},  # No cameras for this test
            calibration_dir=Path(follower_cal_path).parent,
            max_relative_target=3.0,
        )
        
        # Instantiate both classes
        leader = SO101Leader(leader_config)
        follower = SO101Follower(follower_config)
        print('Both SO101Leader and SO101Follower instantiated successfully')
        
        # Connect both
        leader.connect(calibrate=False)
        follower.connect(calibrate=False)
        print('Both devices connected successfully')
        
        target_period = 1.0 / target_freq  # 0.1 seconds
        
        print(f'Running interleaved operations for {duration} seconds at {target_freq}Hz...')
        print('Pattern: Leader Action -> Follower Observation -> repeat')
        
        start_time = time.time()
        loop_count = 0
        timing_stats = {'leader_times': [], 'follower_times': [], 'loop_times': [], 'follower_send_times': []}
        
        while time.time() - start_time < duration:
            loop_start = time.time()
            
            # Get observation from follower
            follower_start = time.time()
            observation = follower.get_observation()
            follower_end = time.time()
            follower_time = (follower_end - follower_start) * 1000  # ms
            timing_stats['follower_times'].append(follower_time)

            # Get action from leader
            leader_start = time.time()
            action = leader.get_action()
            leader_end = time.time()
            leader_time = (leader_end - leader_start) * 1000  # ms
            timing_stats['leader_times'].append(leader_time)

            # use follower observation as the action (don't move the follower)
            follower_send_start = time.time()
            follower.send_action(observation)
            follower_send_end = time.time()
            follower_send_time = (follower_send_end - follower_send_start) * 1000  # ms
            timing_stats['follower_send_times'].append(follower_send_time)
            # time.sleep(0.05)  # 50ms delay to allow bus recovery after write
            
            loop_end = time.time()
            loop_time = (loop_end - loop_start) * 1000  # ms
            timing_stats['loop_times'].append(loop_time)
            
            loop_count += 1
            
            # Print periodic updates
            # if loop_count % 20 == 0:  # Every 2 seconds at 10Hz
            elapsed = time.time() - start_time
            actual_freq = loop_count / elapsed
            print(f'  Loop {loop_count}: {actual_freq:.1f}Hz, '
                    f'Follower: {follower_time:.1f}ms, '
                    f'Leader: {leader_time:.1f}ms, '
                    f'Follower Send: {follower_send_time:.1f}ms, '
                    f'Total: {loop_time:.1f}ms')
            
            # Sleep to maintain 10Hz
            elapsed_loop_time = time.time() - loop_start
            if elapsed_loop_time < target_period:
                print(f'  Sleeping for {target_period - elapsed_loop_time:.1f}s to maintain 10Hz')
                time.sleep(target_period - elapsed_loop_time)
        
        # Calculate statistics
        total_elapsed = time.time() - start_time
        actual_freq = loop_count / total_elapsed
        
        leader_avg = sum(timing_stats['leader_times']) / len(timing_stats['leader_times'])
        follower_avg = sum(timing_stats['follower_times']) / len(timing_stats['follower_times'])
        follower_send_avg = sum(timing_stats['follower_send_times']) / len(timing_stats['follower_send_times'])
        loop_avg = sum(timing_stats['loop_times']) / len(timing_stats['loop_times'])
        
        leader_max = max(timing_stats['leader_times'])
        follower_max = max(timing_stats['follower_times'])
        follower_send_max = max(timing_stats['follower_send_times'])
        loop_max = max(timing_stats['loop_times'])
        
        print(f'\n--- 10Hz Interleaved Test Results ---')
        print(f'Total loops: {loop_count}')
        print(f'Duration: {total_elapsed:.1f}s')
        print(f'Target frequency: {target_freq:.1f}Hz')
        print(f'Actual frequency: {actual_freq:.1f}Hz')
        print(f'Leader get_action() - Avg: {leader_avg:.1f}ms, Max: {leader_max:.1f}ms')
        print(f'Follower get_observation() - Avg: {follower_avg:.1f}ms, Max: {follower_max:.1f}ms')
        print(f'Follower Send: {follower_send_time:.1f}ms, Avg: {follower_send_avg:.1f}ms, Max: {follower_send_max:.1f}ms')
        print(f'Total loop time - Avg: {loop_avg:.1f}ms, Max: {loop_max:.1f}ms')
        
        success = actual_freq >= target_freq * 0.9  # Allow 10% tolerance

        # print('Waiting for 500ms...')
        # time.sleep(0.5)
        
        # Disconnect both
        leader.disconnect()
        follower.disconnect()
        print('Both devices disconnected successfully')
        
        if success:
            print('✅ 10Hz interleaved test PASSED')
        else:
            print('⚠️  10Hz interleaved test completed but frequency was below target')
        
        return success
        
    except Exception as e:
        print(f'10Hz interleaved test FAILED: {e}')
        traceback.print_exc()
        try:
            if 'leader' in locals():
                leader.disconnect()
            if 'follower' in locals():
                follower.disconnect()
        except:
            pass
        return False

def test_follower_send_action_issue(follower_port, follower_cal_path):
    """Test to isolate the send_action issue"""
    print(f'\n=== Testing Follower send_action Issue ===')
    try:
        # Create follower configuration - test both with and without max_relative_target
        follower_config = SO101FollowerConfig(
            id='dum_e_follower',
            port=follower_port,
            use_degrees=False,
            cameras={},
            calibration_dir=Path(follower_cal_path).parent,
            max_relative_target=None  # Start with None to avoid extra read
        )
        
        follower = SO101Follower(follower_config)
        follower.connect(calibrate=False)
        print('Follower connected successfully')
        
        # Get initial observation
        print('Getting initial observation...')
        obs1 = follower.get_observation()
        print(f'Initial observation successful, keys: {list(obs1.keys())}')
        
        # Create a simple action (same positions to avoid movement)
        action = {key: value for key, value in obs1.items() if key.endswith('.pos')}
        print(f'Created action with keys: {list(action.keys())}')
        
        # Test 1: send_action without max_relative_target
        print('\nTest 1: send_action without max_relative_target')
        try:
            sent_action = follower.send_action(action)
            print(f'send_action successful: {list(sent_action.keys())}')
            
            # Try to get observation immediately after
            time.sleep(0.01)  # Small delay
            obs2 = follower.get_observation()
            print('✅ get_observation after send_action: SUCCESS')
            
        except Exception as e:
            print(f'❌ Test 1 FAILED: {e}')
            traceback.print_exc()
        
        # Test 2: Configure with max_relative_target and test again
        print('\nTest 2: Testing with max_relative_target configured')
        follower.disconnect()
        
        follower_config_with_limit = SO101FollowerConfig(
            id='dum_e_follower',
            port=follower_port,
            use_degrees=False,
            cameras={},
            calibration_dir=Path(follower_cal_path).parent,
            max_relative_target=50.0  # This will trigger extra read in send_action
        )
        
        follower_with_limit = SO101Follower(follower_config_with_limit)
        follower_with_limit.connect(calibrate=False)
        
        try:
            obs3 = follower_with_limit.get_observation()
            action2 = {key: value for key, value in obs3.items() if key.endswith('.pos')}
            
            sent_action2 = follower_with_limit.send_action(action2)
            print(f'send_action with limit successful: {list(sent_action2.keys())}')
            
            # Try to get observation immediately after
            time.sleep(0.01)  # Small delay
            obs4 = follower_with_limit.get_observation()
            print('✅ get_observation after send_action with limit: SUCCESS')
            
        except Exception as e:
            print(f'❌ Test 2 FAILED: {e}')
            traceback.print_exc()
        
        follower_with_limit.disconnect()
        return True
        
    except Exception as e:
        print(f'Follower send_action test FAILED: {e}')
        traceback.print_exc()
        return False

def test_minimal_send_read_loop(follower_port, follower_cal_path):
    """Minimal test to isolate the send_action -> get_observation issue"""
    print(f'\n=== Testing Minimal Send-Read Loop ===')
    try:
        follower_config = SO101FollowerConfig(
            id='dum_e_follower',
            port=follower_port,
            use_degrees=False,
            cameras={},
            calibration_dir=Path(follower_cal_path).parent,
            max_relative_target=None  # No extra reads
        )
        
        follower = SO101Follower(follower_config)
        follower.connect(calibrate=False)
        print('Follower connected')
        
        # Get initial state
        obs = follower.get_observation()
        action = {key: value for key, value in obs.items() if key.endswith('.pos')}
        print(f'Initial observation successful')
        
        # Test simple loop: send_action -> get_observation
        for i in range(100):
            print(f'\nIteration {i+1}:')
            
            # Send action (write)
            try:
                start_time = time.time()
                sent = follower.send_action(action)
                print(f'  ✅ send_action successful')
                end_time = time.time()
                dt_follower_send = (end_time - start_time) * 1000  # ms
            except Exception as e:
                print(f'  ❌ send_action failed: {e}')
                break
            
            # Small delay
            time.sleep(1 / 30.0)  # 30Hz delay
            
            # Get observation (read)
            try:
                start_time = time.time()
                obs = follower.get_observation()
                print(f'  ✅ get_observation successful')
                end_time = time.time()
                dt_follower_get = (end_time - start_time) * 1000  # ms
            except Exception as e:
                print(f'  ❌ get_observation failed: {e}')
                traceback.print_exc()
                break
            
            print(f'Loop {i+1}: send_action: {dt_follower_send:.1f}ms, get_observation: {dt_follower_get:.1f}ms')
        
        follower.disconnect()
        return True
        
    except Exception as e:
        print(f'Minimal send-read test FAILED: {e}')
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    # logging.basicConfig(level=logging.INFO)
    init_logging(console_level='DEBUG')
    
    print("SO-101 Individual Motor Testing")
    print("=" * 40)
    
    # Test port scanning
    print("\n1. PORT SCANNING TEST")
    # follower_scan = test_port_scan('/dev/so101_follower')
    # leader_scan = test_port_scan('/dev/so101_leader')
    
    # Test individual motor buses
    print("\n2. MOTOR BUS TESTING")

    follower_cal_path = '/home/melon/sherry/lerobot/calibration/robots/so101_follower/dum_e_follower.json'
    
    # follower_success = test_motor_bus(
    #     '/dev/so101_follower',
    #     follower_cal_path,
    #     'FOLLOWER'
    # )
    
    # leader_success = test_motor_bus(
    #     '/dev/so101_leader', 
    #     '/home/melon/sherry/lerobot/calibration/teleoperators/so101_leader/dum_e_leader.json',
    #     'LEADER'
    # )
    
    # # Test SO101 classes
    # print("\n3. SO101 CLASS TESTING")
    
    # leader_class_success = test_so101_leader(
    #     '/dev/so101_leader',
    #     '/home/melon/sherry/lerobot/calibration/teleoperators/so101_leader/dum_e_leader.json'
    # )
    
    # follower_class_success = test_so101_follower(
    #     '/dev/so101_follower',
    #     follower_cal_path
    # )
    
    # # Test follower send_action issue
    # print("\n4. FOLLOWER SEND_ACTION ISSUE TESTING")
    
    # send_action_success = test_follower_send_action_issue(
    #     '/dev/so101_follower',
    #     follower_cal_path
    # )
    
    # # Test minimal send-read loop
    # print("\n5. MINIMAL SEND-READ LOOP TESTING")
    
    # minimal_loop_success = test_minimal_send_read_loop(
    #     '/dev/so101_follower',
    #     follower_cal_path
    # )
    
    # Test 10Hz interleaved operation
    print("\n6. 10HZ INTERLEAVED TESTING")
    
    interleaved_success = test_interleaved_10hz(
        '/dev/so101_leader',
        '/home/melon/sherry/lerobot/calibration/teleoperators/so101_leader/dum_e_leader.json',
        '/dev/so101_follower',
        follower_cal_path,
        duration=5.0  # Run for 10 seconds
    )
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY:")
    # print(f"Port scanning: {'✅' if follower_scan and leader_scan else '❌'}")
    # print(f"Follower bus: {'✅' if follower_success else '❌'}")
    # print(f"Leader bus: {'✅' if leader_success else '❌'}")
    # print(f"SO101Leader class: {'✅' if leader_class_success else '❌'}")
    # print(f"SO101Follower class: {'✅' if follower_class_success else '❌'}")
    # print(f"Send action issue test: {'✅' if send_action_success else '❌'}")
    # print(f"Minimal send-read loop: {'✅' if minimal_loop_success else '❌'}")
    print(f"10Hz interleaved: {'✅' if interleaved_success else '❌'}")
    
    # all_tests_passed = (follower_success and leader_success and 
    #                    leader_class_success and follower_class_success and
    #                    send_action_success and minimal_loop_success and interleaved_success)
    
    # if all_tests_passed:
    #     print("\n✅ All tests passed!")
    #     print("Motor buses, SO101 classes, and 10Hz operation are all working correctly.")
    # else:
    #     print("\n❌ Some tests failed. Check individual results above.")

if __name__ == '__main__':
    main()
