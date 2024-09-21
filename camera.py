def getMacFrames(device=1):
    import cv2
    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera Feed", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
