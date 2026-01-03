import serial
import math
import time

# ================= SERIAL =================
PORT = "COM3"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(2)
ser.reset_input_buffer()

# ================= THAM SỐ =================
DEADZONE = 3    # độ: dừng xoay
CORRECT_ZONE = 8 # độ: giữ hướng khi chạy
dF = dL = dR = scanL = scanR = 999


T_TO_P0  = 2       # current -> point0
T_0_TO_1 = 3        # point0 -> point1
T_1_TO_2 = 3      # point1 -> point2

# ================= ĐIỀU KHIỂN =================
def quay_phai():  ser.write(b"R\n")
def quay_trai():  ser.write(b"L\n")
def dung():       ser.write(b"S\n")
def chay_thang(): ser.write(b"F\n")

def tinh_bearing(lat_from, lon_from, lat_to, lon_to):
    dy = (lat_to - lat_from) * 111194
    dx = (lon_to - lon_from) * 111194 * math.cos(math.radians(lat_from))
    b = math.degrees(math.atan2(dx, dy))
    return b + 360 if b < 0 else b

def bo_qua_30_mau():
    cnt = 0
    while cnt < 10:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        try:
            float(line)
            cnt += 1
        except ValueError:
            continue

def xoay_toi_huong(bearing_target):
    print(f"\n🎯 XOAY → {bearing_target:.1f}°")

    # 🔴 BỎ 30 MẪU TRƯỚC KHI XOAY
    bo_qua_30_mau()

    while True:
        try:
            heading = float(ser.readline().decode().strip())
        except:
            continue

        alpha = (bearing_target - heading + 180) % 360 - 180
        print(
            f"XOAY | Heading:{heading:6.1f}° | Alpha:{alpha:+6.1f}°",
            end="\r"
        )

        if abs(alpha) < DEADZONE:
            dung()
            print("\n✅ ĐÚNG HƯỚNG")
            break
        elif alpha > 0:
            quay_phai()
        else:
            quay_trai()

        time.sleep(0.05)

def chay_giu_huong(bearing_target, run_time):
    print(f"🚗 CHẠY {run_time}s + GIỮ HƯỚNG")

    # 🔴 BỎ 30 MẪU TRƯỚC KHI CHẠY
    bo_qua_30_mau()

    t0 = time.time()
    while time.time() - t0 < run_time:
        try:
            heading = float(ser.readline().decode().strip())
        except:
            continue

        alpha = (bearing_target - heading + 180) % 360 - 180
        print(
            f"CHẠY | Heading:{heading:6.1f}° | Alpha:{alpha:+6.1f}°",
            end="\r"
        )

        if abs(alpha) <= CORRECT_ZONE:
            chay_thang()
        elif alpha > 2:
            quay_phai()
        elif alpha < -2:
            quay_trai()

        time.sleep(0.05)

    dung()
    print("\n⏹ DỪNG")


# vị trí hiện tại
current_lat = 16.803050
current_lon = 107.103311

# các điểm trên vỉa hè
points = [(16.80305557, 107.10332116), (16.80347684, 107.10308103), (16.80362197, 107.10339928)]

print("\n🚀 BẮT ĐẦU ĐIỀU KHIỂN ROBOT\n")

# current → point 0 
bearing0 = tinh_bearing(current_lat, current_lon, *points[0])
xoay_toi_huong(bearing0)
chay_giu_huong(bearing0, T_TO_P0)

# point 0 → point 1 
bearing1 = tinh_bearing(*points[0], *points[1])
xoay_toi_huong(bearing1)
chay_giu_huong(bearing1, T_0_TO_1)

#  point 1 → point 2 
bearing2 = tinh_bearing(*points[1], *points[2])
xoay_toi_huong(bearing2)
chay_giu_huong(bearing2, T_1_TO_2)

print("\n🏁 HOÀN TẤT LỘ TRÌNH")
