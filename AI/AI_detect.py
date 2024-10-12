from flask import request, Flask, jsonify
import multiprocessing as mp
import cv2
import torch
from detection_model import model_detect
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

rstm_path = []
queues = []
# 接收到的信息
request_data = []


def process_model(processes, model_processes, queue_id, request_data, rstm_path, queues, queue_img):
    # 进程读流和模型启动模块
    if len(rstm_path) % 4 == 0:
        queue_id = len(rstm_path) // 4

    if len(rstm_path) % 4 == 1:

        # model_de = load_model()
        queue = mp.Queue(maxsize=10)
        queues.append(queue)

        model_process = mp.Process(target=model_detect, args=(queues[queue_id - 1], request_data, queue_img ))
        model_process.start()
        model_processes.append(model_process)

    proc = mp.Process(target=read_rstm, args=(rstm_path[-1], queues[queue_id - 1]))
    proc.start()
    processes.append(proc)

    # if len(rstm_path) % 4 == 1:


def requests_data(path_video, id, data, queue_img):
    queue_id = 1
    processes = []
    model_processes = []

    path_jost = data[0]

    path = path_video

    # path = data['rtspUrl']

    rstm_path.append([path, id])
    request_data.append(data)

    process_model(processes, model_processes, queue_id, request_data, rstm_path, queues, queue_img)


# 加载模型
def load_model():
    model = torch.hub.load('/', 'custom', path=r'D:\wendang\yolov5-maste2\spy_5.10.pt',
                           source='local')
    return model


# 进程读流
def read_rstm(rstm, queue):
    cap = cv2.VideoCapture(rstm[0])
    if not cap.isOpened():
        print(f"Error opening video stream: {rstm[0]}")
        return
    while True:
        ret, frame = cap.read()
        print(rstm)
        if not ret:
            break
        # 调整帧的大小以减少内存使用
        frame = cv2.resize(frame, (640, 360))
        # 根据需要调整大小
        # 限制队列大小
        if queue.qsize() < 10:
            queue.put((rstm[0], rstm[1], frame))
    cap.release()


# 检测照片和坐标
queue_img = mp.Queue(maxsize=20)

@app.route('/detect/AI', methods=['POST'])
def run_request():
    # data 里面的视频流地址给file

    data = request.get_json()
    path_jost = data[0]
    path = path_jost['rtspUrl']
    id = path_jost['id']

    requests_data(path, id, data, queue_img)

    return jsonify({'status': 'ok'})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

