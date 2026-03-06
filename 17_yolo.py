def Train():

    from ultralytics import YOLO


    MODELs = YOLO('yolov5n.pt') 
    results = MODELs.train(data= r'D:\..\data.yaml',
                        epochs=10 , batch=8, imgsz=320, verbose=False)

# عمل الاهختبار عن طريق الكامرا
def TestCam():
    from ultralytics import YOLO
    import cv2

    # best.pt ملف الـ 
    model = YOLO(r'C:\..\best.pt')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error could not read frame')
            break

        results = model.predict(frame)

        annotated_frame = results[0].plot()
        cv2.imshow('yolo camera detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


TestCam()


