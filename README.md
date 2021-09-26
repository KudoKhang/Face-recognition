# Face Recogniton
Công nghệ nhận diện khuôn mặt đã xuất hiện trong cuộc sống xung quanh ta cũng đã hơn 10 năm. Ban đầu là trên camera của điện thoại thông minh, xác định khuôn mặt để lấy nét chính xác hơn. Hay gần đây hơn là hệ thống tự tag bạn bè khi bạn up ảnh lên facebook. Điều đáng kinh ngạch là công nghệ này có độ chính xác rất cao, khoảng 98% - gần tương đương với khả năng nhận diện của con người.
Dự án nhận diện khuôn mặt là một trong những dự án thú vị trong Thị giác máy tính (Computer Vision)

<p align="center">
	<img src="https://images.viblo.asia/ae9af402-43d5-4e9c-8af9-9c23f00ae622.jpeg" />
</p>

# How it works
Ở dự án này mình sẽ sử dụng thư viện `face_recognition`  (Thư viện này được giới thiệu là "**Nhận dạng khuôn mặt bằng những dòng code python ngắn nhất thế giới**", "**Được xây dựng bằng cách sử dụng tính năng hiện đại nhất của Dlib, độ tiêu chuẩn chính xác đến 99,38% với những khuôn mặt được gắn nhãn trong tiêu chuẩn Wild**"). Ngoài ra mình còn dùng `OpenCV` một thư viện rất quen thuộc trong Thị giác máy tính để xây dựng một ứng dụng có thể nhận diện khuôn mặt từ camera theo thời gian thực.
Ở bước cuối cùng sẽ thêm 1 tính năng tuy ngắn gọn nhưng rất thú vị đó chính là ghi lại mốc thời gian và tên người đã được cung cấp từ trước vào 1 file CSV. Như vậy chúng ta có thể xây dựng một ứng dụng chấm công cho công nhân hoặc điểm danh học sinh bằng khuôn mặt khá là đơn giản!
Bây giờ là các bước chính để thực hiện dự án này và một chút thông tin cách mà `face_recognition` xử lý ở phần backend.

## Bước 1: Tìm tất cả khuôn mặt
Tấc nhiên rồi, để nhận diện xem khuôn mặt đó là ai thì bước đầu tiên chúng ta sẽ phải tìm kiếm tất cả khuôn mặt xuất hiện trong khung hình. 

<p align="center">
	<img src="https://www.researchgate.net/profile/Luis-Piardi/publication/338941941/figure/fig3/AS:854434024284166@1580724354772/Face-detection-using-the-HOG-algorithm.ppm"/>
</p>

Phương pháp được sử dụng ở đây có tên là HOG (Histogram of Oriented Gradients - Tạm dịch là biểu đồ của hướng dốc). Ý tưởng chính của phương pháp này để tìm khuôn mặt là ta chuyển ảnh qua đen trắng (vì màu sắc không ảnh hưởng tới việc nhận diện nên ta có thể bỏ qua), sau đó ta nhìn vào từng điểm ảnh và so sánh với các điểm ảnh lân cận, mục đích là tìm ra hướng thay đổi sáng hay tối của khu vực đó. Như vậy ta có thể chuyển ảnh gốc sang tập hợp các hướng sáng, thể hiện cấu trúc đơn giản của khuôn mặt. Để tìm ra khuôn mặt trong hình ảnh **HOG**, chúng ta cần tìm phần ảnh có cấu trúc giống nhất với cấu trúc HOG được trích lọc từ trong quá trình đào tạo. Sử dụng kỹ thuật này, chúng ta có thể dễ dàng tìm ra khuôn mặt ở bất cứ hình ảnh nào.

Tuy nhiên, trong `face_recognition ` ta có thể thực hiện việc đó một các dễ dàng:

```python
import face_recognition
image = face_recognition.load_image_file("obama.jpg")
face_locations = face_recognition.face_locations(image)
```

<p align="center">
	<img src="https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png" />
</p>

## Bước 2: Chỉnh hướng khuôn mặt
Khi chụp ảnh thì khuôn mặt không phải lúc nào cũng hướng thẳng vào camera, chính vì thế khi trích xuất khuôn mặt từ bước 1, để mô hình thực hiện nhận diện tốt nhất, ta cần phải chỉnh hướng khuôn mặt sao cho khuôn mặt được đặt vào chính giữa. 
Chúng ta thực hiện điều đó với thuật toán **face landmark estimation** (ước lượng cột mốc trên mặt), được tạo ra bởi Vahid Kazemi và Josephine vào năm 2014. 
Ý tưởng của thuật toán này là chúng ta xác định 68 điểm cột mốc trên khuôn mặt, điều này sẽ được thực hiện bằng cách huấn luyện một thuật toán máy học. Sau khi có các điểm đó rồi ta sẽ thực hiện các phép biến đổi như quay, phóng to, thu nhỏ hay cắt ảnh để có được mắt và môi nằm ở giữa trung tâm nhất có thể

<p align="center">
	<img src="https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png" />
</p>

- Lưu ý: Không sử dụng các phương pháp biến đổi 3D vì như thế sẽ gây biến dạng khuôn mặt.

## Bước 3: So sánh khuôn mặt
Khi đã trích xuất được 2 khuôn mặt rồi, làm sao để máy tính có thể so sánh và đưa ra kết luận 2 khuôn mặt đó có phải là của một người hay không? Để giải quyết vấn đề này chúng ta sử dụng **Deep Convolutional Neural Network** - mạng nơnon tích chập đa lớp để nhận diện 128 điểm đo lương trên khuôn mặt. 

```python
img = face_recognition.load_image_file('obama.jpg')
imgEncoding = face_recognition.face_encodings()
```

Công việc cuối cùng là so sánh 128 điểm đã được trích xuất trên, nếu độ chênh lệch của các điểm giữa khuôn mặt được mã hoá là thấp thì chúng ta kết luận được đó là cùng một người.

## Bước 4: Đánh dấu thời gian xuất hiện
Qua 3 bước trên, chúng ta đã hiểu cách `face_recognition` thực hiện việc nhận diện khuôn mặt. Bây giờ chúng ta chỉ việc một hàm để chương trình lưu lại tên và thời gian người đó xuất hiện vào file csv nữa là xong:

```python
def markAttendance(name):  
	 with open('Attendance.csv', 'r+') as f:  
		 myDataList = f.readlines()  
		 nameList = []  
		 for line in myDataList:  
			 entry = line.split(',')  
			 nameList.append(entry[0])  
		 if name not in nameList:  
			 now = datetime.now()  
			 dtStr = now.strftime('%H:%M:%S')  
			 f.writelines(f'\n{name}, {dtStr}')
```

<p align="center">
    <img src="https://i.imgur.com/h9ANt2W.png" />
</p>

Kết quả cuối cùng, ta được một project nhận diện khuôn mặt như sau: 

<p align="center">
	<img src="result.gif" />
</p>

# Installation
Để cài đặt các thư viện cần thiết bạn chỉ cần cài đặt `requirements.txt`. Tui nhiên tuỳ hệ điều hành mà `cmake`  sẽ được cài đặt khác nhau

```
pip install requirements.txt
```

# Usage
Để sử dụng, chúng ta clone repo về máy và khởi chạy:

```
git clone https://github.com/KudoKhang/Face-recognition
cd Face-recognition
python AttendanceProject.py
```

