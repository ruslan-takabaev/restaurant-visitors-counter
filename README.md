### 1 - Download project files and model weights
```
$git clone https://github.com/ruslan-takabaev/restaurant-visitors-counter.git && cd restaurant-visitors-counter
$mkdir -p 'rec/raw' 'rec/annotated' 'faces/' 'model/' && cd model/
```
Download and move the yolo model to 'model/' subdirectory. Model: https://www.kaggle.com/models/ruslantakabaev/yolov8s-people-and-faces

### 2 - Python environment setup
```
$python3 -m venv <desired_venv_name>
$source <desired_venv_name>/bin/activate
$pip install -r requirements.txt
```
It is possible that I missed a few packages when writing requirements.txt, so update it if needed.

