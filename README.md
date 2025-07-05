# Handwritten Digit Recognition (Draw-on-Screen)

This project lets you draw digits on the screen using your mouse and predicts them using a CNN trained on the MNIST dataset.

## Files
- `train_model.py`: Trains and saves the digit recognition model.
- `draw_digit.py`: Launches an OpenCV canvas to draw and predict digits.
  
## How to Use
1. Run `train_model.py` to generate `digit_model.h5`.
2. Run `draw_digit.py`, draw digits with your mouse.
3. Press `P` to predict, `R` to reset, `Q` to quit.

## Requirements
- Python 3.8–3.11
- TensorFlow
- OpenCV
