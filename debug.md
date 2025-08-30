# SO-101 Teleoperation Debugging Session

## Summary
Debugging communication issues between LeRobot's teleoperation system and SO-101 robot hardware with STS3215 Feetech servos.

## Initial Problem
Teleoperation command failed with motor communication errors:
```bash
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/so101_follower \
    --robot.id=dum_e_follower \
    --robot.calibration_dir=/home/melon/sherry/so101_bench/bringup/calibration/robots/so101_follower \
    --robot.camera_configs_path=/home/melon/sherry/so101_bench/bringup/camera_configs.json \
    --teleop.type=so101_leader \
    --teleop.port=/dev/so101_leader \
    --teleop.id=dum_e_leader \
    --teleop.calibration_dir=/home/melon/sherry/so101_bench/bringup/calibration/teleoperators/so101_leader \
    --display_data=true
```

**Error Messages:**
- `[TxRxResult] There is no status packet!`
- `[TxRxResult] Incorrect status packet!`
- `Failed to write 'Lock' on id_=2`
- `Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6]`

## Investigation Steps

### 1. Hardware vs Software Issue
- **Initial hypothesis:** Hardware problems with motors
- **Reality:** Motors work perfectly with bambot.org/feetech.js software
- **Conclusion:** Software configuration issue, not hardware

### 2. Motor Detection Test
Ran port scanning to verify motor connectivity:
```python
from lerobot.motors.feetech.feetech import FeetechMotorsBus
scan_result = FeetechMotorsBus.scan_port('/dev/so101_follower')
```

**Results:**
- ✅ All 6 motors detected on both ports
- ✅ Correct model numbers (777 = STS3215)  
- ✅ Communication at 1,000,000 baud rate
- ✅ Motor IDs 1-6 responding to ping

### 3. Protocol Investigation
- Checked baudrate settings (1M baud is correct)
- Investigated timeout and retry settings
- Compared leader vs follower arm behavior
- Found both arms have identical hardware setup

### 4. Calibration Path Discovery
**Root Cause Found:** Wrong calibration directory!

- **Original paths:** `/home/melon/sherry/so101_bench/bringup/calibration/`
- **Correct paths:** `/home/melon/sherry/lerobot/calibration/`

### 5. Individual Motor Bus Testing
Created test script that confirmed both buses work independently:
- ✅ Follower motors: Read positions successfully
- ✅ Leader motors: Read positions successfully  
- ✅ Both use same STS3215 configuration
- ✅ Calibration loading works correctly

### 6. Concurrent Operation Issue
**Final Issue:** Resource contention during simultaneous operation

- Individual buses work perfectly ✅
- Teleoperation (simultaneous leader + follower) fails ✗
- Position data shows brief success before sync_read failures
- Likely USB bandwidth or motor bus timing conflict

## Solutions Applied

### 1. Correct Calibration Paths
```bash
--robot.calibration_dir=/home/melon/sherry/lerobot/calibration/robots/so101_follower
--teleop.calibration_dir=/home/melon/sherry/lerobot/calibration/teleoperators/so101_leader
```

### 2. Reduced Frame Rate
Changed from 60 FPS to 15 FPS to match 30 FPS cameras and reduce timing conflicts:
```bash
--fps=15
```

## Current Status
- ✅ Motor hardware confirmed working
- ✅ Individual motor buses working
- ✅ Calibration loading resolved
- ✅ Teleoperation starts successfully
- ⚠️ Intermittent sync_read failures during concurrent operation

## Remaining Issue
**Resource Contention:** When both leader and follower motor buses operate simultaneously, communication becomes unreliable after initial success. This suggests:

1. **USB bandwidth limitations**
2. **Motor bus timing conflicts**  
3. **Need for staggered read operations**
4. **Potential retry logic improvements**

## Next Steps
1. Implement staggered motor reads between leader/follower
2. Reduce retry counts to prevent blocking
3. Add better error handling in teleoperation loop
4. Consider lower overall frame rates for stability

## Key Files
- Motor test script: `test_individual_motors.py`
- Calibration files: `/home/melon/sherry/lerobot/calibration/`
- Camera configs: `/home/melon/sherry/so101_bench/bringup/camera_configs.json`