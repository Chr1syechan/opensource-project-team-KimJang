# Virtual-pad-using-webcam

yolov5와 opencv, roboflow, colab을 이용한 가상패드를 구현하여 이미지파일을 만들기

colab으로 yolov5의 train.py 를 이용하여 roboflow에서 알맞은 학습주소를 찾은 후 머신러닝

머신러닝한 best.pt를 yolov5로 객체인식 실행

객체의 좌표를 실시간으로 받으며 opencv, imshow 모듈로 화면에 그려 보여줌

보여진 화면을 저장시켜줌

## Requirements

```bash
yolov5==v7.0==AGPL-3.0 License
opencv-python==4.8.1==3-clause BSD License
```

### Install

```bash
pip install -r requirements.txt
pip install opencv-python torch
```

### On terminal

```bash
python run.py
```
