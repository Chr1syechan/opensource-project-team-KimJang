import cv2
import torch
import numpy as np

# YOLOv5 모델 및 특정 체크포인트 (best.pt) 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_final.pt', force_reload=True) # Autoshaping을 통해 다양한 이미지 크기를 처리

# 모델을 평가 모드로 설정
model.eval()

# 카메라 열기 (일반적으로 0은 내장 웹캠을 나타냄)
cap = cv2.VideoCapture(0)

path=[]
d = False
hand = True

while cap.isOpened():
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 객체 검출
    with torch.no_grad():
        results = model(frame)

    # 결과 가져오기
    pred = results.pred[0]

    if d:
        if hand:
            for det in pred:
                cls, bbox = int(det[5].item()), det[:4].tolist()
                x1, y1, x2, y2 = map(int, bbox)  # bbox는 [x_min, y_min, x_max, y_max] 형식의 좌표를 가지고 있음

                if model.names[cls] == 'Pen':
                    path.append((x2, y2))
        elif not hand:
            for det in pred:
                cls, bbox = int(det[5].item()), det[:4].tolist()
                x1, y1, x2, y2 = map(int, bbox)  # bbox는 [x_min, y_min, x_max, y_max] 형식의 좌표를 가지고 있음

                if model.names[cls] == 'Pen':
                    path.append((x1, y2))

    for i in range(1, len(path)):
        cv2.line(frame, path[i - 1], path[i], (0, 255, 0), 2)

    cv2.imshow('pen tracking', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite('detected_objects.jpg', frame)
        print("이미지 저장 완료.")

    elif key == ord('d'):
        if d:
            d = False
            print("작성정지")
        else:
            d = True
            print("작성시작")
    elif key == ord('r'):
        hand = True
        print("오른손잡이")
    elif key == ord('l'):
        hand = False
        print("왼손잡이")
    # 'q' 키를 눌러 종료
    elif key == ord('q'):
        break

# 카메라 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
