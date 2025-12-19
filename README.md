这个项目是将mediapipe的疲劳检测方法移植到-瑞芯微-国产化芯片RK系列开发板上

当前这个版本使用的瑞芯微开发板是RK3568
检测类型包括：

---就座状态

---疲劳/异常状态:闭眼、打哈欠（张嘴）、歪头/视线偏离

main-facelib-phone-rknn.py中额外使用yolo添加了手机检测，当前版本yolo只完成了rknn模型的转换，并未放到NPU上跑，因此可能会有点慢。



部署方式：

与开发板通信后，创建虚拟环境python=3.8。

然后pip项目中的requirements，安装所需的库，库中已经包含了rk-toolkit-lite2，如果下载不了或者不能用requirement安装，可以自己下载rk-toolkit-lite2.whl并进行安装。

安装好后，连接摄像头，查找摄像头设备号（如：video（0），不一定所有人都是这个，我的摄像头被识别到了video（13），使用前提前查一下摄像头）。


<img width="227" height="85" alt="cap" src="https://github.com/user-attachments/assets/38841c0c-a9c3-4fde-a29d-075c42c8168f" />

走完上面的流程就可以直接跑了。

使用情况：

1.正常状态——normal


<img width="636" height="441" alt="normal" src="https://github.com/user-attachments/assets/86367d54-d3b2-4a4f-acfb-05e2047cad67" />


2.闭眼状态——eye


<img width="634" height="437" alt="eye" src="https://github.com/user-attachments/assets/d504ce60-aff0-445a-b690-b3cc22b75212" />

3.打哈欠状态——mouth


<img width="636" height="440" alt="mouth" src="https://github.com/user-attachments/assets/eb822a89-fb5c-4701-8c6d-3ec0235b99c6" />


4.歪头注意力不集中状态——head



<img width="637" height="439" alt="head" src="https://github.com/user-attachments/assets/2178d2b2-5bd8-49b5-83bd-a1481a2fed1f" />



以上截图是在mobax上，使用x service投影到pc上截的图，但是在rk3568上跑，受x service的传输影响，所以帧率显示5帧，板子上理论可以跑到10帧+，基本流畅，rk3588上实测可达到30帧+。











