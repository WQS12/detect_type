import cv2
from multiprocessing import Queue, Process
import time

def read_rtsp(input_queue, rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error opening video stream: {rtsp_url}")
        return
    skip = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if skip % 3 == 0:
            input_queue.put(frame)
        skip += 1
        time.sleep(0.06)
    input_queue.put(None)
    cap.release()


def main(rtsp):
    input_queue = Queue(maxsize=30)
    output_queue = Queue(maxsize=30)
    p = Process(target=read_rtsp, args=(input_queue, rtsp))

    p.start()
    while True:
        data = output_queue.get()
        if data is None:
            break
        frame,r,t,l,b = data
    p.join()

if __name__ == '__main__':
    processes = []
    for i in range(32):
        A = Process(target=main, args=(r"./demo.mp4",))
        A.start()
        print(f"Starting detection for {i}")
        processes.append(A)
        time.sleep(0.05)

    print(len(processes))
    # 等待所有进程完成
    for p in processes:
        p.join()




