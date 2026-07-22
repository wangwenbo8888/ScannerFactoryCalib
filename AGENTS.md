# 开发规则

## 核心原则：有现成的就照搬，不自作聪明

**当存在已验证可工作的实现时，原样照搬，不要发明替代方案。**

### 背景

`E:\workfold\20260509intergrate\LeadScanK2\series` 是扫描仪相机的**已验证稳定实现**（采集、停止、再采集均正常出图）。factory_calib 在移植它的功能时，必须原样照搬其代码结构和调用方式，不要"参考着改"。

### 具体要求

1. **照搬，不要"参考"**：照搬 = 逐字复制关键代码 + 保持相同的调用顺序和对象生命周期。"参考着改"= 只借鉴思路、自己重新组织代码 —— 后者已被多次证明会引入隐蔽 bug，浪费时间。

2. **Galaxy 相机采集**（已踩坑，务必照搬 LeadScanK2 `CameraControl.cpp`）：
   - `start_ScannerAcquisition`：每次都 `GetRemoteFeatureControl` + `OpenStream`（二者跨 stop 不可复用），`RegisterCaptureCallback` + `StartGrab`，配置 `Line2` 触发，**两台相机都配完后再分别 `AcquisitionStart`**。
   - `close_ScannerAcquisition`：`AcquisitionStop` → `StopGrab` → `UnregisterCaptureCallback` → delete handler → `Close` stream。

3. **预览显示**（已踩坑，务必照搬 LeadScanK2 `LEADSCANSeries.cpp`）：
   - 回调线程调 `setLeftImage`/`setRightImage`，**左右按 frameID 配对**后 `cv::resize(0.25)` → `convertMattoQImage`（`Format_Indexed8` + 逐行 memcpy 到 scanLine + 256 灰度颜色表）→ `emit updateImages(QPixmap::fromImage(...).toImage(), QPixmap::fromImage(...).toImage())`。
   - `connect(updateImages, onUpdateImages, Qt::QueuedConnection)`。
   - `onUpdateImages`：`label->setPixmap(QPixmap::fromImage(img))`，纯 `QLabel`。
   - **不要**用：单张独立 emit、`PreviewWidget` 子类、`convertToFormat`、自己发明的 metatype 跨线程投递。

4. **改动前先读现成实现**：动手前先读 LeadScanK2 对应函数的完整代码，确认每一行，再照搬。不要凭记忆。

### 已踩过的坑（不要再踩）

- **scanner 启动按钮里先 `stop`(N11) 再 `start`(N10) 连续发送 → MCU 偶发不启动**（第一次稳定、后续偶发不出图/不出帧的根因）。照搬 LeadScanK2：`on_pushButton_Start_Scanner_Clicked` 直接发 N10，**不先 stop**。停再启由用户分别点「停止」「启动」（秒级间隔），不在一次点击里连发 N11+N10。
- stream 跨 stop/start 复用 → 第二次采集不出帧
- QImage 跨线程用 `convertToFormat` / 单张 emit → 偶发不显示
- `PreviewWidget` 子类 + 自定义 styleSheet → 渲染偶发失效
- AcquisitionStart 时机（左先右后）→ 触发 race
- `CameraFrame::frameIndex` 用默认 0 而非真实 `GetFrameID()` → setLeftImage 配对错误（初始 m_iRightId=0==m_iLeftId=0），resize 空 Mat 崩溃

这些 LeadScanK2 都已正确处理，照搬即可避免。
