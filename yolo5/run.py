import cv2
import torch

# YOLOv5 모델 및 특정 체크포인트 (best_final.pt) 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_final.pt', force_reload=True) # Autoshaping을 통해 다양한 이미지 크기를 처리

# 모델을 평가 모드로 설정
model.eval()

# 카메라 열기 (일반적으로 0은 내장 웹캠을 나타냄)
cap = cv2.VideoCapture(0)

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

    # 결과를 화면에 표시 및 저장할 이미지에 경로 그리기
    img_with_boxes = frame.copy()
    for det in pred:
        conf, cls, bbox = det[4].item(), int(det[5].item()), det[:4].tolist()
        if conf > 0.5:  # 신뢰도가 0.5 이상인 경우에만 표시
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f'x1 : {x1:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 검출된 객체가 표시된 프레임 보기
    cv2.imshow('YOLOv5 객체 검출', img_with_boxes)

    # 's' 키를 눌러 현재 화면을 이미지로 저장
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('detected_objects.jpg', img_with_boxes)
        print("이미지 저장 완료.")

    # 'q' 키를 눌러 종료
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
