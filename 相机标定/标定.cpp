#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

int main()
{
    // 设置标定板的尺寸和格点数（例如，棋盘格）
    cv::Size patternSize(9, 6);  // (列数, 行数)
    float squareSize = 0.0254f;  // 格点实际尺寸（单位：米）

    // 创建标定板对象
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(patternSize.width, patternSize.height, squareSize, squareSize * 0.5f);

    // 初始化标定板角点容器
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<std::vector<cv::Point3f>> objectPoints;

    // 配置相机
    rs2::config config;
    rs2::pipeline pipeline;
    config.enable_stream(rs2_stream::RS2_STREAM_COLOR, 640, 480, rs2_format::RS2_FORMAT_BGR8, 30);  // 设置彩色图像流

    // 启动相机并进行标定
    pipeline.start(config);
    while (true)
    {
        // 等待新帧
        rs2::frameset frames = pipeline.wait_for_frames();
        rs2::frame colorFrame = frames.get_color_frame();
        if (!colorFrame)
            continue;

        // 将图像转换为OpenCV格式
        cv::Mat colorImage(cv::Size(colorFrame.as<rs2::video_frame>().get_width(), colorFrame.as<rs2::video_frame>().get_height()), CV_8UC3, (void*)colorFrame.get_data(), cv::Mat::AUTO_STEP);

        // 检测标定板角点
        std::vector<cv::Point2f> corners;
        bool found = cv::aruco::detectMarkers(colorImage, board->dictionary, corners, board->ids);
        if (found)
        {
            // 绘制标定板角点
            cv::aruco::drawDetectedMarkers(colorImage, corners);

            // 提取二维角点
            std::vector<cv::Point2f> charucoCorners;
            cv::aruco::interpolateCornersCharuco(corners, board, colorImage, charucoCorners);
            if (charucoCorners.size() > 0)
            {
                imagePoints.push_back(charucoCorners);

                // 提取三维角点
                std::vector<cv::Point3f> charucoObjectPoints;
                cv::aruco::getBoardObjectAndImagePoints(board, charucoCorners, corners, charucoObjectPoints);
                objectPoints.push_back(charucoObjectPoints);
            }
        }

        // 显示图像
        cv::imshow("Calibration", colorImage);
        if (cv::waitKey(1) == 'q')
            break;
    }

    // 进行相机标定
    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    cv::calibrateCamera(objectPoints, imagePoints, cv::Size(colorFrame.as<rs2::video_frame>().get_width(), colorFrame.as<rs2::video_frame>().get_height()), cameraMatrix, distCoeffs, rvecs, tvecs);

    // 打印标定结果
    std::cout << "Camera matrix:" << std::endl;
    std::cout << cameraMatrix << std::endl;
    std::cout << "Distortion coefficients:" << std::endl;
    std::cout << distCoeffs << std::endl;

    // 关闭相机
    pipeline.stop();
    cv::destroyAllWindows();

    return 0;
}
