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
DEADZONE = 3          # độ: dừng xoay
CORRECT_ZONE = 7   # độ: giữ hướng khi chạy

dF = dL = dR = scanL = scanR = 999

T_TO_P0  = 15
T_0_TO_1 = 6
T_1_TO_2 = 6

# ================= ĐIỀU KHIỂN =================
def quay_phai():  ser.write(b"R\n")
def quay_trai():  ser.write(b"L\n")
def dung():       ser.write(b"S\n")
def chay_thang(): ser.write(b"F\n")
def quay_servo():  ser.write(b"V\n")
def clear_scan():  ser.write(b"C\n")

# ================= BEARING =================
def tinh_bearing(lat_from, lon_from, lat_to, lon_to):
    dy = (lat_to - lat_from) * 111194
    dx = (lon_to - lon_from) * 111194 * math.cos(math.radians(lat_from))
    b = math.degrees(math.atan2(dx, dy))
    return b + 360 if b < 0 else b

# ================= BỎ MẪU NHIỄU =================
def bo_qua_mau(n=10):
    cnt = 0
    while cnt < n:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        try:
            float(line)
            cnt += 1
        except:
            continue

def xoay_toi_huong(bearing_target):
    print(f"\n🎯 XOAY → {bearing_target:.1f}°")

    # 🔴 BỎ 30 MẪU TRƯỚC KHI XOAY
    bo_qua_mau(10)

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

def chay_thang_time(run_time):
    t0 = time.time()
    while time.time() - t0 < run_time:
        chay_thang()
        time.sleep(0.05)
    dung()  

def ne_s_phai(h):
    print("\n↪️ NÉ S PHẢI")
    xoay_toi_huong((h + 70) % 360)
    chay_thang_time(2)
    xoay_toi_huong(h)


def ne_s_trai(h):
    print("\n↩️ NÉ S TRÁI")
    xoay_toi_huong((h - 70) % 360)
    chay_thang_time(2)
    xoay_toi_huong(h)

    


# ================= CHẠY GIỮ HƯỚNG =================
def chay_giu_huong(bearing_target, run_time):
    print(f"\n🚗 CHẠY {run_time}s + GIỮ HƯỚNG")
    bo_qua_mau(10)

    t0 = time.time()
    heading = None
    scan_requested = False
    time_obstacle = None  # Lưu thời điểm gặp vật cản
    total_run_time = run_time  # Lưu thời gian chạy ban đầu

    while time.time() - t0 < run_time:
        elapsed = time.time() - t0
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue

        # ===== HEADING =====
        if not line.startswith("U:"):
            try:
                heading = float(line)
            except:
                pass
            continue

        # ===== ULTRASONIC =====
        if heading is None:
            continue

        try:
            _, data = line.split(":")
            parts = data.split(",")
            if len(parts) != 5:
                continue
            dF, dL, dR, scanL, scanR = map(int, parts)
        except:
            continue

        alpha = (bearing_target - heading + 180) % 360 - 180

        print(
            f"CHẠY | Time:{elapsed:.1f}s | Heading:{heading:6.1f}° | Alpha:{alpha:+6.1f}° | dF={dF:3d}",
            end="\r"
        )
        
        # Phát hiện vật cản
        if dF < 15 and scan_requested == False:
            print(f"\n⚠️  VẬT CẢN ở {elapsed:.1f}s - NÉ NGAY")
            dung()
            quay_servo()
            time.sleep(0.5)
            scan_requested = True

            time_obstacle = elapsed  # Ghi nhận thời điểm gặp vật cản
            print(f"ScanL={scanL}, ScanR={scanR} | dL={dL}, dR={dR}")
            
            if scanR < scanL:
                ne_s_phai(heading)
            else: 
                ne_s_trai(heading)
            
            clear_scan()
            
            # Quay lại hướng gốc (bearing_target) để chạy tiếp
            xoay_toi_huong(bearing_target)
            
            # Reset timer: không tính thời gian né
            remaining_time = total_run_time - time_obstacle  # Thời gian còn lại cần chạy
            t0 = time.time()  # Reset timer gốc
            run_time = remaining_time  # Set run_time = thời gian còn lại cần chạy
            print(f"📌 Gặp vật cản lúc {time_obstacle:.1f}s → Chạy tiếp {remaining_time:.1f}s nữa")
            scan_requested = False
            continue

        if abs(alpha) <= CORRECT_ZONE:
            chay_thang()
        elif alpha > 3:
            quay_phai()
        elif alpha < -3:
            quay_trai()

        time.sleep(0.05)

    dung()
    print("\n⏹ DỪNG")

# ================= LỘ TRÌNH =================
current_lat = 16.803050
current_lon = 107.103311

points = [
    (16.80305557, 107.10332116),
    (16.80344509, 107.10309784),
    (16.80333406, 107.10285678)
]

print("\n🚀 BẮT ĐẦU ĐIỀU KHIỂN ROBOT\n")

# current → point 0
bearing0 = tinh_bearing(current_lat, current_lon, *points[0])
xoay_toi_huong(bearing0)
chay_giu_huong(bearing0, T_TO_P0)

# point 0 → point 1
bearing1 = tinh_bearing(*points[0], *points[1])
xoay_toi_huong(bearing1)
chay_giu_huong(bearing1, T_0_TO_1)

# point 1 → point 2
bearing2 = tinh_bearing(*points[1], *points[2]) 
xoay_toi_huong(bearing2)
chay_giu_huong(bearing2, T_1_TO_2)

print("\n🏁 HOÀN TẤT LỘ TRÌNH")
