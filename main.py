import cv2
from multiprocessing import Queue, Process, Pipe
import time

def receive_data(queues):
    data = []
    while len(data) < 8:
        for queue in queues:
            if len(data) < 8:  # 继续检查，直到 data 长度达到 8
                try:
                    item = queue.get_nowait()  # 非阻塞地获取队列中的数据
                    data.append(item)
                except Exception:
                    continue  # 如果队列为空，继续下一个队列
            else:
                break  # 如果 data 长度已达到 8，则退出循环
    return data
def model_detect(data):

    return data
def read_rtsp(id,input_queues,output_queues,rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error opening video stream: {rtsp_url}")
        return
    skip = 0
    index = id % 8
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if skip % 3 == 0:
            input_queues[index].put([id,frame])
        skip += 1
        #---------------------------------------------------------------------
        #数据传回对应读流进程（演示功能)
        try:
            data = output_queues[index].get_nowait()  # 非阻塞地获取队列中的数据
        except Exception:
            continue  # 如果队列为空，继续下一个队列
        
        print('数据来源:',data[0],'ID:',id)
        time.sleep(0.06)
    input_queues[index].put(None)
    cap.release()


def main(input_queues,output_queues):
    while True:
        data = receive_data(input_queues)#取流
        results = model_detect(data)#模型检测
        #--------------------
        #根据ID发送数据
        for result in results:
            index = result[0] % 8 
            output_queues[index].put(result)
        
    

if __name__ == '__main__':
    video_processes = []
    main_process = []
    input_queues = []
    output_queues = []
    queue_id = -1#队列分组索引
    for i in range(30):
        if i % 8 == 0:
            queue_id += 1
        #---------------------------------------------------------------
        #动态增加队列组
            input_queues.append([Queue(30) for _ in range(8)])
            output_queues.append([Queue(30) for _ in range(8)])
        #---------------------------------------------------------------
        #数据中转（对应8路读流）
            A = Process(target=main, args=(input_queues[queue_id],output_queues[queue_id]))
            A.start()
            print(f"Starting detection for {i}")
            main_process.append(A)
        #---------------------------------------------------------------
        #读流
        A = Process(target=read_rtsp, args=(i,input_queues[queue_id],output_queues[queue_id],r"./demo.mp4"))
        A.start()
        print(f"Starting detection for {i}")
        video_processes.append(A)
        time.sleep(0.05)

    print(len(video_processes))
    # 等待所有进程完成
    for p in video_processes:
        p.join()
    for p in main_process:
        p.join()
