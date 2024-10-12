import torch
import cv2
model = torch.hub.load('/', 'custom', path=r'D:\wendang\yolov5-maste2\spy_5.10.pt',
                       source='local')
def model_detect(queue, data, queue_img):
    # 检测区域坐标
    batch = []
    frame_coordinat = {}
    frame_img = {}
    paths = []
    # 触发时间
    frame_num = 0
    # 触发间隔时间
    interval_time = 1200

    try:
        while True:
            if not queue.empty():
                rstm, id, frame = queue.get()
                if rstm not in paths:
                    paths.append(rstm)

                batch.append([id, frame])

                if len(batch) == 4:
                    frame_num += 1

                    results = model([item[1] for item in batch])

                    print('reslut:',results)
                    for i in range(len(results)):
                        frame_img[batch[i][0]] = batch[i][1]

                        # 初始化该帧的坐标列表
                        frame_coordinat[batch[i][0]] = []


                        for index, row in results.pandas().xyxy[i].iterrows():
                            x, y, w, h = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                            confidence = row['confidence']
                            class_id = row['name']

                            image_np = batch[i][1]
                            frame_img[batch[i][0]] = image_np
                            # 将每个目标的坐标信息添加到该帧的坐标列表中

                            frame_coordinat[batch[i][0]].append([x, y, w, h, confidence, class_id])

                    # 事件检测函数

                    queue_img.put([frame_img, frame_coordinat])
                    # print('img_coor:', queue_img.get())
                    batch.clear()

    except KeyboardInterrupt:
        print("Program interrupted by user.")
        cv2.destroyAllWindows()