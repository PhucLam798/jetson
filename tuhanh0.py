import serial
import math
import time

# ================= SERIAL CONFIG =================
PORT = "/dev/ttyUSB0"     # change to ttyACM0 if needed
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=0.3)
time.sleep(2)
ser.reset_input_buffer()

# ================= PARAMETERS =================
DEADZONE = 3.0
CORRECT_ZONE = 8.0

T_TO_P0  = 2.0
T_0_TO_1 = 3.0
T_1_TO_2 = 3.0

# ================= MOTOR COMMANDS =================
def turn_right():
    ser.write(b"R\n")

def turn_left():
    ser.write(b"L\n")

def stop():
    ser.write(b"S\n")

def forward():
    ser.write(b"F\n")

# ================= SERIAL READ =================
def read_heading():
    """Read compass heading, ignore ultrasonic lines"""
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        if line.startswith("U:"):
            continue
        try:
            return float(line)
        except:
            continue

def flush_heading(n=10):
    """Discard noisy heading samples"""
    cnt = 0
    while cnt < n:
        line = ser.readline().decode(errors="ignore").strip()
        try:
            float(line)
            cnt += 1
        except:
            continue

# ================= BEARING =================
def calc_bearing(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111194.0
    dx = (lon2 - lon1) * 111194.0 * math.cos(math.radians(lat1))
    b = math.degrees(math.atan2(dx, dy))
    return b + 360.0 if b < 0 else b

# ================= ROTATE =================
def rotate_to(bearing_target):
    print(f"\nRotate to {bearing_target:.1f} deg")
    flush_heading(10)

    while True:
        heading = read_heading()
        alpha = (bearing_target - heading + 180) % 360 - 180

        print(
            f"ROTATE | Heading:{heading:6.1f} | Error:{alpha:+6.1f}",
            end="\r"
        )

        if abs(alpha) < DEADZONE:
            stop()
            print("\nAligned")
            break
        elif alpha > 0:
            turn_right()
        else:
            turn_left()

        time.sleep(0.07)

# ================= DRIVE WITH HEADING HOLD =================
def drive_hold_heading(bearing_target, run_time):
    print(f"\nDrive {run_time:.1f}s with heading hold")
    flush_heading(10)

    t0 = time.time()
    while time.time() - t0 < run_time:
        heading = read_heading()
        alpha = (bearing_target - heading + 180) % 360 - 180

        print(
            f"DRIVE | Heading:{heading:6.1f} | Error:{alpha:+6.1f}",
            end="\r"
        )

        if abs(alpha) <= CORRECT_ZONE:
            forward()
        elif alpha > 0:
            turn_right()
        else:
            turn_left()

        time.sleep(0.07)

    stop()
    print("\nStop")

# ================= ROUTE =================
current_lat = 16.803050
current_lon = 107.103311

points = [
    (16.80305557, 107.10332116),
    (16.80347684, 107.10308103),
    (16.80362197, 107.10339928),
]

print("\n=== AUTONOMOUS START ===")

# current -> point 0
bearing0 = calc_bearing(current_lat, current_lon, *points[0])
rotate_to(bearing0)
drive_hold_heading(bearing0, T_TO_P0)

# point 0 -> point 1
bearing1 = calc_bearing(*points[0], *points[1])
rotate_to(bearing1)
drive_hold_heading(bearing1, T_0_TO_1)

# point 1 -> point 2
bearing2 = calc_bearing(*points[1], *points[2])
rotate_to(bearing2)
drive_hold_heading(bearing2, T_1_TO_2)

print("\n=== ROUTE COMPLETE ===")
