#!/bin/bash

# Camera Identification Script
# This script helps identify USB cameras and their unique identifiers for udev rules

echo "=== USB Camera Discovery Tool ==="
echo "This script will help you identify your cameras for udev rule creation"
echo ""

# Check if running as root (needed for some udev operations)
if [[ $EUID -eq 0 ]]; then
    echo "⚠️  Running as root - good for complete device info"
else
    echo "ℹ️  Running as user - some device info may be limited"
    echo "   Consider running with 'sudo ./identify_cameras.sh' for complete info"
fi

echo ""
echo "=== Step 1: List all video devices ==="
echo ""

# Find all video devices
# ls /dev/video* → tries to list any file whose name starts with video in /dev (e.g., /dev/video0, /dev/video1). 
# These are Video4Linux device nodes—cameras, capture cards, etc.
# 2>/dev/null → redirects stderr (file descriptor 2) to /dev/null, which discards it.
# That way, if there are no /dev/video* devices, ls’s “No such file or directory” error won’t clutter your output.
# sort -V → “version sort”, meaning it treats numbers inside strings as numbers rather than plain text.
video_devices=($(ls /dev/video* 2>/dev/null | sort -V))

if [ ${#video_devices[@]} -eq 0 ]; then
    echo "❌ No video devices found!"
    echo "   Make sure your cameras are connected and detected by the kernel"
    echo "   Try: dmesg | grep -i video"
    exit 1
fi

echo "Found ${#video_devices[@]} video device(s):"
for device in "${video_devices[@]}"; do
    echo "  $device"
done

echo ""
echo "=== Step 2: Identify actual cameras (not metadata devices) ==="
echo ""

actual_cameras=()
for device in "${video_devices[@]}"; do
    # Check if this is an actual capture device (not metadata)
    caps=$(v4l2-ctl --device="$device" --list-formats-ext 2>/dev/null | grep -c "Video Capture" || echo "0")
    
    if [ "$caps" -gt 0 ]; then
        actual_cameras+=("$device")
        echo "📹 $device - Camera (capture device)"
        
        # Get basic camera info
        camera_name=$(v4l2-ctl --device="$device" --info 2>/dev/null | grep "Card type" | cut -d':' -f2- | xargs)
        driver=$(v4l2-ctl --device="$device" --info 2>/dev/null | grep "Driver name" | cut -d':' -f2- | xargs)
        
        echo "   Name: $camera_name"
        echo "   Driver: $driver"
    else
        echo "🔧 $device - Metadata/other device"
    fi
done

if [ ${#actual_cameras[@]} -eq 0 ]; then
    echo "❌ No actual camera devices found!"
    exit 1
fi

echo ""
echo "=== Step 3: Get USB device information for each camera ==="
echo ""

# creates an associative array (also called a hash map or dictionary) in bash
declare -A camera_info

for i in "${!actual_cameras[@]}"; do
    device="${actual_cameras[i]}"
    echo "Camera $((i+1)): $device"
    echo "----------------------------------------"
    
    # Get the device path in sysfs
    device_path=$(udevadm info --query=path --name="$device" 2>/dev/null)
    
    # -z string = true if string is empty (zero length)
    if [ -z "$device_path" ]; then
        echo "⚠️  Could not get device path for $device"
        continue
    fi
    
    # Get udev properties
    echo "Getting device properties..."
    
    # Key identifiers we need
    vendor_id=$(udevadm info --query=property --path="$device_path" | grep "ID_VENDOR_ID=" | cut -d'=' -f2)
    product_id=$(udevadm info --query=property --path="$device_path" | grep "ID_MODEL_ID=" | cut -d'=' -f2)
    serial=$(udevadm info --query=property --path="$device_path" | grep "ID_SERIAL_SHORT=" | cut -d'=' -f2)
    usb_path=$(udevadm info --query=property --path="$device_path" | grep "ID_PATH=" | cut -d'=' -f2)
    devpath=$(udevadm info --query=property --path="$device_path" | grep "DEVPATH=" | cut -d'=' -f2)
    
    echo "  Vendor ID: ${vendor_id:-'Not found'}"
    echo "  Product ID: ${product_id:-'Not found'}"
    echo "  Serial Number: ${serial:-'Not found'}"
    echo "  USB Path: ${usb_path:-'Not found'}"
    echo "  Device Path: ${devpath:-'Not found'}"
    
    # Store for later use
    camera_info["$device,vendor"]="$vendor_id"
    camera_info["$device,product"]="$product_id"
    camera_info["$device,serial"]="$serial"
    camera_info["$device,path"]="$usb_path"
    
    # Get physical USB port info if available
    echo ""
    echo "  USB Port Information:"
    if [ -n "$devpath" ]; then
        # Extract USB bus and port info from devpath
        usb_bus_port=$(echo "$devpath" | grep -o 'usb[0-9]\+/[0-9-]\+\(\.[0-9]\+\)*' | head -1)
        if [ -n "$usb_bus_port" ]; then
            echo "    USB Bus/Port: $usb_bus_port"
        fi
        
        # Try to get USB hub port
        port_num=$(echo "$devpath" | grep -o '\.[0-9]\+:' | tail -1 | tr -d '.:')
        if [ -n "$port_num" ]; then
            echo "    Physical Port: $port_num"
        fi
    fi
    
    echo ""
done

echo "=== Step 4: Analysis and Recommendations ==="
echo ""

# Group cameras by USB path to handle duplicates from same physical camera
declare -A path_to_devices
declare -A unique_cameras

echo "Grouping cameras by USB path..."
for device in "${actual_cameras[@]}"; do
    path="${camera_info["$device,path"]}"
    if [ -n "$path" ] && [ "$path" != "Not found" ]; then
        if [ -z "${path_to_devices["$path"]}" ]; then
            path_to_devices["$path"]="$device"
        else
            path_to_devices["$path"]="${path_to_devices["$path"]} $device"
        fi
    else
        # If no path info, treat as unique
        unique_cameras["$device"]="$device"
    fi
done

# Select one device per USB path (prefer lower numbered /dev/video devices)
final_cameras=()
echo ""
for path in "${!path_to_devices[@]}"; do
    devices_str="${path_to_devices["$path"]}"
    read -ra devices_array <<< "$devices_str"
    
    # Sort devices and pick the first one (lowest /dev/video number)
    selected_device=$(printf '%s\n' "${devices_array[@]}" | sort | head -1)
    final_cameras+=("$selected_device")
    
    echo "USB Path: $path"
    echo "  Found devices: ${devices_array[*]}"
    echo "  Selected: $selected_device"
    echo ""
done

# Add any cameras without path info
for device in "${!unique_cameras[@]}"; do
    final_cameras+=("$device")
    echo "No USB path info for $device - including as unique camera"
done

echo "Final camera list after deduplication: ${final_cameras[*]}"
echo ""

# Check if we have exactly 3 cameras after deduplication
if [ ${#final_cameras[@]} -ne 3 ]; then
    echo "⚠️  Expected 3 cameras after deduplication, found ${#final_cameras[@]}"
    echo "   This script is designed for exactly 3 cameras"
fi

# Update actual_cameras to use deduplicated list
actual_cameras=("${final_cameras[@]}")

# Analyze what we can use to distinguish cameras
echo "Distinguishing characteristics available:"

# Check if serial numbers are available and unique
serials=()
for device in "${actual_cameras[@]}"; do
    serial="${camera_info["$device,serial"]}"
    if [ -n "$serial" ] && [ "$serial" != "Not found" ]; then
        serials+=("$serial")
    fi
done

if [ ${#serials[@]} -eq 3 ] && [ "${serials[0]}" != "${serials[1]}" ] && [ "${serials[0]}" != "${serials[2]}" ] && [ "${serials[1]}" != "${serials[2]}" ]; then
    echo "✅ Serial numbers: Available and unique - RECOMMENDED"
    use_serials=true
else
    echo "❌ Serial numbers: Not available or not unique"
    use_serials=false
fi

# Check USB paths (should now be unique after deduplication)
paths=()
for device in "${actual_cameras[@]}"; do
    path="${camera_info["$device,path"]}"
    if [ -n "$path" ] && [ "$path" != "Not found" ]; then
        paths+=("$path")
    fi
done

if [ ${#paths[@]} -eq 3 ] && [ "${paths[0]}" != "${paths[1]}" ] && [ "${paths[0]}" != "${paths[2]}" ] && [ "${paths[1]}" != "${paths[2]}" ]; then
    echo "✅ USB paths: Available and unique - Good fallback"
    use_paths=true
else
    echo "❌ USB paths: Not available or not unique"
    use_paths=false
fi

echo ""
echo "=== Step 5: Suggested udev rules ==="
echo ""

if [ "$use_serials" = true ]; then
    echo "Based on serial numbers (recommended):"
    echo ""
    for i in "${!actual_cameras[@]}"; do
        device="${actual_cameras[i]}"
        serial="${camera_info["$device,serial"]}"
        vendor="${camera_info["$device,vendor"]}"
        product="${camera_info["$device,product"]}"
        
        if [ $i -eq 0 ]; then
            cam_name="cam_top"
        elif [ $i -eq 1 ]; then
            cam_name="cam_front"
        else
            cam_name="cam_wrist"
        fi
        
        echo "# Camera $((i+1)): $device -> /dev/$cam_name"
        echo "SUBSYSTEM==\"video4linux\", ATTRS{idVendor}==\"$vendor\", ATTRS{idProduct}==\"$product\", ATTRS{serial}==\"$serial\", SYMLINK+=\"$cam_name\""
        echo ""
    done
elif [ "$use_paths" = true ]; then
    echo "Based on USB paths (physical port-based):"
    echo ""
    for i in "${!actual_cameras[@]}"; do
        device="${actual_cameras[i]}"
        path="${camera_info["$device,path"]}"
        vendor="${camera_info["$device,vendor"]}"
        product="${camera_info["$device,product"]}"
        
        if [ $i -eq 0 ]; then
            cam_name="cam_top"
        elif [ $i -eq 1 ]; then
            cam_name="cam_front"
        else
            cam_name="cam_wrist"
        fi
        
        echo "# Camera $((i+1)): $device -> /dev/$cam_name"
        echo "SUBSYSTEM==\"video4linux\", ATTRS{idVendor}==\"$vendor\", ATTRS{idProduct}==\"$product\", ENV{ID_PATH}==\"$path\", SYMLINK+=\"$cam_name\""
        echo ""
    done
else
    echo "⚠️  Cannot create reliable udev rules - cameras are not distinguishable"
    echo "   You may need to use different USB ports or check camera firmware"
fi

echo ""
echo "=== Next Steps ==="
echo "1. Review the suggested udev rules above"
echo "2. Decide which camera should be 'top', 'front', and 'wrist'"
echo "3. You may need to physically swap cameras or modify the rules"
echo "4. Copy the rules to /etc/udev/rules.d/99-cameras.rules"
echo "5. Run: sudo udevadm control --reload-rules && sudo udevadm trigger"
echo "6. Test with the cam_check.sh script"

echo ""
echo "=== Raw udevadm output for reference ==="
echo "(You can use this for manual rule creation if needed)"
echo ""

for device in "${actual_cameras[@]}"; do
    echo "--- $device ---"
    device_path=$(udevadm info --query=path --name="$device" 2>/dev/null)
    if [ -n "$device_path" ]; then
        udevadm info --query=property --path="$device_path" | grep -E "(ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL|ID_PATH|DEVPATH)" || echo "No relevant properties found"
    fi
    echo ""
done

# Common causes of camera not being picked up:
# 1. Camera is not connected to a high-speed USB port (USB 2.0 or USB 3.x). 
# To check speed of each port: lsusb -t
# You should see at least 480M (USB 2.0)
