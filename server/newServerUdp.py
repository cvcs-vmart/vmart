import cv2

if __name__ == '__main__':

    capture = cv2.VideoCapture('udp://@0.0.0.0:12345')
    if not capture.isOpened():
        capture = cv2.VideoCapture('udp://@0.0.0.0:12345')
        print("Cannot open camera")

    while True:
        ret, frame = capture.read()
        if not ret:  # fine dello stream
            break
        color = cv2.cvtColor(frame, cv2.IMREAD_COLOR_BGR)
        cv2.imshow('myframe', color)
        print('Fatto')
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
