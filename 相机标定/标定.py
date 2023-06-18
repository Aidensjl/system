import pyrealsense2 as rs
import numpy as np

# 设置标定板的尺寸和格点数（例如，棋盘格）
pattern_size = (9, 6)  # (列数, 行数)
square_size = 0.0254  # 格点实际尺寸（单位：米）

# 创建标定板对象
pattern = rs.pattern.create_checkerboard_pattern(pattern_size[0], pattern_size[1], square_size)

# 初始化标定板角点容器
obj_points = []  # 物体坐标系中的三维点
img_points = []  # 图像坐标系中的二维点

# 配置相机
config = rs.config()
pipeline = rs.pipeline()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)  # 设置彩色图像流

# 启动相机并进行标定
pipeline.start(config)
try:
    while True:
        # 等待新帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 将图像转换为OpenCV格式
        color_image = np.asanyarray(color_frame.get_data())

        # 检测标定板角点
        found, corners = rs.pattern.find_pattern(color_image, pattern)
        if found:
            # 绘制标定板角点
            rs.draw_frame(color_image, corners)

            # 提取二维角点
            img_points.append(corners)

            # 提取三维角点
            obj_pts = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            obj_pts[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
            obj_points.append(obj_pts)

        # 显示图像
        cv2.imshow('Calibration', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 进行相机标定
    _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(obj_points, img_points, color_frame.get_width(),
                                                              color_frame.get_height(), None, None)

    # 打印标定结果
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(dist_coeffs)

finally:
    # 关闭相机
    pipeline.stop()
    cv2.destroyAllWindows()
