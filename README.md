# -mediapipe-
这个项目是将mediapipe的疲劳检测方法移植到国产化芯片RK系列开发板上

当前这个版本使用的RK开发板是RK3568
检测类型包括：
---就座状态
---疲劳/异常状态:闭眼、打哈欠（张嘴）、歪头/视线偏离

main-facelib-phone-rknn.py中额外使用yolo添加了手机检测，当前版本yolo只完成了rknn模型的转换，并未放到NPU上跑，因此可能会有点慢

