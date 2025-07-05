import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("digit_model.h5")

canvas = np.zeros((400, 400), dtype="uint8")
drawing = False
ix, iy = -1, -1

def draw(event, x, y, flags, param):
    global drawing, ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, (ix, iy), (x, y), 255, 20)  # smoother lines
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("Draw Digit (Press P to Predict, R to Reset, Q to Quit)")
cv2.setMouseCallback("Draw Digit (Press P to Predict, R to Reset, Q to Quit)", draw)

while True:
    cv2.imshow("Draw Digit (Press P to Predict, R to Reset, Q to Quit)", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("p"):
        digit = cv2.resize(canvas, (28, 28))
        digit = digit.reshape(1, 28, 28, 1).astype("float32") / 255
        pred = model.predict(digit, verbose=0)
        print("Predicted Digit:", np.argmax(pred))

    elif key == ord("r"):
        canvas[:] = 0

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
