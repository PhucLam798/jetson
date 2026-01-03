import cv2
import torch
import numpy as np
import time


def init_midas(model_type: str = "MiDaS_small"):
    """Khởi tạo model MiDaS và transform tương ứng."""
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    return midas, transform, device


def compute_depth_map(frame, midas, transform, device):
    """Tính depth map từ một frame BGR."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    return depth_map


def draw_three_vertical_zones(depth_map):
    """Chia depth map thành 3 cột dọc bằng nhau, vẽ và trả về depth_color, mean từng cột."""
    # Chuẩn hóa để hiển thị
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = np.uint8(depth_norm)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    H, W = depth_map.shape
    w3 = W // 3

    zone_means = []

    for j in range(3):
        x1 = j * w3
        # Cột cuối cùng ăn hết phần còn lại để phủ full chiều ngang
        x2 = (j + 1) * w3 if j < 2 else W
        y1, y2 = 0, H

        zone = depth_map[y1:y2, x1:x2]
        zone_means.append(zone.mean())

        cv2.rectangle(depth_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx = x1 + (x2 - x1) // 2
        cy = H // 2
        cv2.circle(depth_color, (cx, cy), 6, (0, 0, 255), -1)

    return depth_color, zone_means


def decide_avoidance_action(zone_means, depth_map):
    """Quyết định hướng né vật cản dựa trên 3 ô dọc.

    zone_means: list/array 3 phần tử [trái, giữa, phải] là giá trị depth trung bình mỗi ô.
    depth_map: toàn bộ depth map của frame hiện tại.

    Ngưỡng phát hiện vật cản được chọn là giá trị trung bình cộng
    của TẤT CẢ điểm ảnh trong depth_map.

    - MiDaS: giá trị càng lớn càng gần.
    - Ô nào có mean > mean_toan_anh được coi là có vật cản.

    Trả về: "F", "L", "R" hoặc None.
    """

    zone_means = np.asarray(zone_means, dtype=float)

    global_mean = float(depth_map.mean())

    # Một ô được coi là có vật cản nếu mean của ô > mean toàn ảnh
    obstacles = zone_means > global_mean
    left_obstacle, mid_obstacle, right_obstacle = obstacles

    # Logic né vật cản:
    # - Nếu ô giữa KHÔNG có vật cản -> đi thẳng "F"
    if not mid_obstacle:
        return "F"

    # - Nếu ô giữa có vật cản:
    #   + Nếu bên trái không có -> rẽ trái "L"
    if not left_obstacle:
        return "L"

    #   + Nếu bên phải không có -> rẽ phải "R"
    if not right_obstacle:
        return "R"

    #   + Nếu cả trái & phải đều có vật cản -> không có hướng né rõ ràng
    return None


def main():
    # 1. Load model & transform
    midas, transform, device = init_midas()

    # 2. Load video/camera
    input_video_path = 0
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Không mở được video!")
        return

    print("Đang chạy depth + chia 3 cột dọc...")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Tính depth map
        depth_map = compute_depth_map(frame, midas, transform, device)

        # 4. Chia 3 cột dọc bằng nhau
        depth_color, zone_means = draw_three_vertical_zones(depth_map)

        print("\n--- 3 vùng dọc (trái -> phải) ---")
        print(np.array(zone_means))

        # 4.1. Quyết định hướng né vật cản (so với ngưỡng là mean toàn ảnh)
        action = decide_avoidance_action(zone_means, depth_map)
        print("Hành động né vật cản:", action)

        # Chuẩn bị text hiển thị hướng đi
        if action is None:
            direction_text = "Dir: NONE"
        else:
            direction_text = f"Dir: {action}"

        # 5. Tính FPS
        curr_time = time.time()
        dt = curr_time - prev_time
        fps = 1.0 / dt if dt > 0 else 0.0
        prev_time = curr_time

        fps_text = f"FPS: {fps:.2f}"

        # Vẽ FPS và hướng đi lên ảnh gốc và ảnh depth
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(depth_color, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, direction_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(depth_color, direction_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2, cv2.LINE_AA)

        # 6. Hiển thị song song ảnh gốc và depth
        h_d, w_d, _ = depth_color.shape
        frame_resized = cv2.resize(frame, (w_d, h_d))
        combined = np.hstack((frame_resized, depth_color))

        cv2.imshow("Real (left) + Depth 3 vertical zones (right)", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
