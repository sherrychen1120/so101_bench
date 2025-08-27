import cv2
import subprocess

cap = cv2.VideoCapture("/dev/cam_top", cv2.CAP_V4L2)  # omit CAP_V4L2 on Windows/macOS

def run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        return e.output.decode("utf-8", errors="replace")

config_name_to_cap_prop = {
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "gain": cv2.CAP_PROP_GAIN,
    "auto_white_balance": cv2.CAP_PROP_AUTO_WB,
    "white_balance_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
    # optional
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "sharpness": cv2.CAP_PROP_SHARPNESS,
    "zoom_absolute": cv2.CAP_PROP_ZOOM,
}

# power line frequency is not available in OpenCV

desired_vals = {
    # 0: Auto Mode
    # 1: Manual Mode
    # 2: Shutter Priority Mode
    # 3: Aperture Priority Mode
    # changing to manual
    'auto_exposure': 1.0,
    'exposure': 100.0, 
    'gain': 5.0, 
    # changing to manual (False)
    'auto_white_balance': 0.0, 
    'white_balance_temperature': 4650.0, 
    # fixed at False
    'autofocus': 0.0,
    # optional
    'brightness': 128.0, 
    'contrast': 128.0, 
    'saturation': 128.0, 
    'sharpness': 128.0, 
    'zoom_absolute': 10.0
}

for key, value in desired_vals.items():
    curr_val = cap.get(config_name_to_cap_prop[key])
    cap.set(config_name_to_cap_prop[key], value)
    print(f"{key}: {curr_val} -> {value}")

print(desired_vals)

print(run(["v4l2-ctl", "-d", "/dev/cam_top", "--all"]))

