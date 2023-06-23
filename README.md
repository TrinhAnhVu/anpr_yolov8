# NGHIÊN CỨU NHẬN DIỆN BIỂN SỐ XE SỬ DỤNG THUẬT TOÁN VÀ MÔ HÌNH YOLO
Áp dụng thuật toán nhận diện đối tượng YOLOv8 + easyOCR vào bài toán nhận diện biến số xe trong hình ảnh hoặc video từ camera giám sát giao thông. 
Các hình ảnh hoặc video này có thể được chụp ở các điểm kiểm soát, giao lộ, bãi đỗ xe, và các đoạn đường khác trong hệ thống giao thông.
-	Từ ảnh đầu vào phát hiện được vùng biển số xe và đọc chính xác ký tự
trên biển số
-	Xây dựng giao diện với các chức năng chọn ảnh(video/camera), nhận diện vùng chứa biển số trong ảnh(video/camera),
hiển thị ký tự đọc được trên biển số.

<details>
  <summary>Cài đặt</summary>
  
- Pip cài đặt gói ultralytics bao gồm tất cả  requirements trong môi trường Python>=3.7 với PyTorch>=1.7 .
```python
pip install ultralytics
```
- Pip cài đặt gói PyQt5
```python
pip install PyQt5      
```
- Pip cài đặt gói easyOCR
```python
pip install easyOCR      
```
</details>

<details>
  <summary>Custom dataset</summary>

- Pip cài đặt roboflow
```python
pip install roboflow
```
- 
</details>
