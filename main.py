import cv2
import numpy as np
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

#IMG WITHOUT BACKGROUND
img = cv2.imread('images/fruit.jpg')
hh, ww = img.shape[:2]

lower = np.array([200, 200, 200])
upper = np.array([255, 255, 255])

thresh = cv2.inRange(img, lower, upper)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
mask = 255 - morph
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# IMG READING
img = cv2.imread('images/mm.jpg')
cv2.imshow('Original superburgir', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# THRESHOLD
ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# WHITE/BLACK
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray img', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# FACE DETECTOR
face = cv2.CascadeClassifier('face.xml')
result = face.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
for (x, y, w, h) in result:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# BLUR
for (x, y, w, h) in result:
    faces = img[y:y + h, x:x + w]
    blur_face = cv2.GaussianBlur(faces, (39, 39), 0)
    img[y:y + h, x:x + w, :] = blur_face
cv2.imshow("Blur Face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ROI
ROI = img[100:500, 50:800]
cv2.imshow('ROI', ROI)
cv2.waitKey(0)
cv2.destroyAllWindows()

# CONTOURE
cont, hir = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cont_img = cv2.drawContours(img.copy(), cont, -1, (0, 255, 0), 2)
cv2.imshow('CONTOUR', cont_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ADD FOR IMG
cv2.rectangle(img, (10, 10), (100, 100), (119, 201, 105), thickness=5)
cv2.line(img, (0, img.shape[0]//2), (img.shape[0], img.shape[1]//2), (119, 201, 105), thickness=3)
cv2.circle(img, (img.shape[0]//2, img.shape[1]//2), 100, (119, 201, 105), thickness=2)
cv2.putText(img, 'mne 1 burgir please', (100, 150), cv2.FONT_HERSHEY_TRIPLEX, 2, (128, 0, 128), 2)
cv2.imshow('img with rectangle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# MORPHOLOGICAL OPERATIONS
# Dilation
kernel = np.ones((8, 8), np.uint8)
dilated_img = cv2.dilate(gray_img, kernel, iterations=1)
cv2.imshow('Dilated Image', dilated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Erosion
kernel = np.ones((8, 8), np.uint8)
eroded_img = cv2.erode(gray_img, kernel, iterations=1)
cv2.imshow('Eroded Image', eroded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Opening
kernel = np.ones((8, 8), np.uint8)
opened_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opened Image', opened_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Closing
kernel = np.ones((8, 8), np.uint8)
closed_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closed Image', closed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Morphological Gradient
kernel = np.ones((5, 5), np.uint8)
gradient_img = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('Morphological Gradient', gradient_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Top Hat
kernel = np.ones((5, 5), np.uint8)
tophat_img = cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('Top Hat', tophat_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Black Hat
kernel = np.ones((5, 5), np.uint8)
blackhat_img = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('Black Hat', blackhat_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# VIDEO READING
stream = cv2.VideoCapture('video/tggr.mp4 ')

num_frames = stream.get(cv2.CAP_PROP_FRAME_COUNT)
frame_ids = np.random.uniform(size=20) * num_frames
frames = []
for fid in frame_ids:
    stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = stream.read()
    if not ret:
        print("SOMETHING WENT WRONG")
        exit()
    frames.append(frame)

median = np.median(frames, axis=0).astype(np.uint8)
median = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

fps = stream.get(cv2.CAP_PROP_FPS)
width = int(stream.get(3))
height = int(stream.get(4))


output = cv2.VideoWriter("video/1",
            cv2.VideoWriter_fourcc('m', 'p', 'g', '4'),
            fps=fps, frameSize=(width, height))

stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    success, frame = stream.read()
    cv2.imshow('video', frame)
    if not success:
        break

    # OBJECT TRACKING
    result = model.track(frame, persist=True)
    frame_ = result[0].plot()
    cv2.imshow('frame', frame_)
    # WHITE/BLACK
    gray_cap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray video', gray_cap)
    # CONTOURE
    cont, hir = cv2.findContours(gray_cap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_cap = cv2.drawContours(frame.copy(), cont, -1, (0, 255, 0), 2)
    cv2.imshow('Video CONTOUR', cont_cap)
    # WITHOUT BACKGROUND
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dif_frame = cv2.absdiff(median, frame)
    threshold, diff = cv2.threshold(dif_frame, 50, 255,
                                    cv2.THRESH_BINARY)
    output.write(diff)
    cv2.imshow("Without background", diff)
    cv2.waitKey(10)
    if cv2.waitKey(1) == ord('q'):
            break


