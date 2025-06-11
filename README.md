## 1 - Download project files and model weights
```
$ git clone https://github.com/ruslan-takabaev/restaurant-visitors-counter.git && cd restaurant-visitors-counter
$ mkdir -p 'rec/raw' 'rec/annotated' 'faces/' 'model/'
```
Download and move the yolo model to 'model/' subdirectory. Model available on [kaggle](https://www.kaggle.com/models/ruslantakabaev/yolov8s-people-and-faces) and [huggingface](https://huggingface.co/rta2101/yolov8s-faces-and-people/blob/main/yolov8s_custom_v1.1.pt)

OR
```
$ cd model/
$ wget https://huggingface.co/rta2101/yolov8s-faces-and-people/resolve/main/yolov8s_custom_v1.1.pt
```


## 2 - Python environment setup
```
$ python3 -m venv <desired_venv_name>
$ source <desired_venv_name>/bin/activate
$ pip install -r requirements.txt
```
It is possible that I missed a few packages when writing requirements.txt, so update it if needed.


## 3 - Run the program
First, open config.py in any text editor and update the ROOT_DIR variable to store the absolute path to the directory where project files are located. After that, update CAMERA_ADDRESS variable to store your actual remote camera address. Update other parameters if needed. 

To run the program:
```
(venv_name)$ python3 app.py
```
If you want to test the websocket streaming, open app_demo.html while app.py is running. Make sure you have the correct IP (use localhost or 127.0.0.1 if running on a local machine) and port (must be same with WEBSOCKET_PORT in config.py) in line 103:
```
    const wsUrl = `ws://${window.location.hostname || 'localhost'}:8765`; // Change this to actual IP and port
```

You can also find documentation (.pdf and .md) on how to access the resources from websocket to build a web app on it.

## 4 - License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/ruslan-takabaev/restaurant-visitors-counter/blob/main/license.txt) file for details.

## 5 - Author
Ruslan Takabaev - [GitHub](https://github.com/ruslan-takabaev)
