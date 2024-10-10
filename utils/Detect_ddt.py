import time
import cv2
import uuid
import os
import ffmpeg
import numpy as np
from datetime import datetime
import pymysql
from pymysql import Error


class DDT:
    def __init__(self):
        pass

    def box_detect(self,frame, list, process_data, det, current_time):
        time_since_last_count_reset = current_time - process_data['motorcycle_event']['last_count_reset_time']
        if time_since_last_count_reset > det['triggerTime']:
            self.reset_count(process_data['motorcycle_event'])

        for i in list:
            if any(polygon.contains(i[0]) for polygon in det["polygons"]):
                process_data['motorcycle_event']['count'] += 1

        if process_data['motorcycle_event']['count'] > 15:
            time_elapsed_since_last_recording = current_time - process_data['motorcycle_event']['last_recording_time']
            if time_elapsed_since_last_recording > det['alarmInhibitionTime']:
                process_data['motorcycle_event']['last_recording_time'] = current_time
                process_data['motorcycle_event']['extra_frames_count'] = 10
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                for i in list:
                    cv2.rectangle(frame, (int(i[1]), int(i[2])), (int(i[3]), int(i[4])), (0, 0, 255), 1)
                path = '/resources/' + str(uuid.uuid4()) + '_2''.jpg'
                cv2.imwrite(path, frame)
                process_data['motorcycle_event']['img_path'] = path

    # 重置计数
    def reset_count(self,data):
        data['count'] = 0
        data['last_count_reset_time'] = time.time()

    # 重置事件
    def reset_event(self,current_time, data):
        self.reset_count(data)
        data['last_count_reset_time'] = current_time
        data['recording'] = False

    # 保存视频
    def save_videos(self,data, description, process_box, img_path):
        camer_Num = data['cameraNum']
        camera_name = data['cameraName']
        event_classify = data['detectionClassify']
        detect_type = data['detectionType']
        frame_queue_box = process_box
        eventLevel = data['alarmLevel']

        formatted_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        video_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        date_path = datetime.now().strftime("%Y/%m/%d")
        base_path = "/home/test/path/"
        full_path = os.path.join(base_path, date_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        video_box_path = fr"{full_path}/{video_time}_{camer_Num}_box_{detect_type}.mp4"
        data_img_path = fr"/profile{img_path}"

        width, height, channels = frame_queue_box[0].shape[1], frame_queue_box[0].shape[0], frame_queue_box[0].shape[2]
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', r=25)  # 假设帧率为30
            .output(video_box_path, pix_fmt='yuv420p', vcodec='libx264', r=30)
            .overwrite_output()
            .run_async(pipe_stdin=True, cmd='/root/ffmpeg_/bin/ffmpeg_custom')
        )
        try:
            for frame in frame_queue_box:
                if frame is not None:
                    process.stdin.write(frame.astype(np.uint8).tobytes())
            process.stdin.close()
            process.wait()

        except Exception as e:
            print(f"录制过程中出现错误: {e}")
        finally:
            print(f"视频转换完成：{video_box_path}")

        video_box_path = fr"/profile{video_box_path}"

        # insert_event_data(camer_Num,camera_name,formatted_datetime, event_classify, detect_type, data_img_path, video_box_path, video_box_path,description, eventLevel)

    # 数据写入txt
    def write_event_data_to_txt(self,camer_Num, camera_name, formatted_datetime, event_classify, detect_type, data_img_path,
                                video_box_path, description, eventLevel):
        # 文件路径
        file_path = "data.txt"
        if not os.path.exists(file_path):
            open(file_path, 'w').close()  # 创建空文件

        with open(file_path, 'a') as file:
            file.write(f"Camera Number: {camer_Num}\n")
            file.write(f"Camera Name: {camera_name}\n")
            file.write(f"Date and Time: {formatted_datetime}\n")  # Assumes already formatted
            file.write(f"Event Classify: {event_classify}\n")
            file.write(f"Detection Type: {detect_type}\n")
            file.write(f"Image Path: {data_img_path}\n")
            file.write(f"Video Path: {video_box_path}\n")
            file.write(f"Description: {description}\n")
            file.write(f"Event Level: {eventLevel}\n")
            file.write("----------\n")
        print("数据写入txt成功")

    # 数据写入数据库
    def insert_event_data(self,camer_Num, camera_name, formatted_datetime, event_classify, detect_type, data_img_path,
                          eventVideoFramed, eventVideo, description, eventLevel, eventState=1):
        event_data = {
            'camera_num': f'{camer_Num}',
            'camera_name': f'{camera_name}',
            'event_time': f'{formatted_datetime}',
            'event_classify': f'{event_classify}',
            'event_type': f'{detect_type}',
            'event_location': f'{camera_name}',
            'event_level': f'{eventLevel}',
            'event_state': f'{eventState}',
            'event_picture': f'{data_img_path}',
            'event_video_framed': f'{eventVideoFramed}',
            'event_video': f'{eventVideo}',
            'event_desc': f'{description}'
        }

        try:
            connection = pymysql.connect(
                host='127.0.0.1',
                port=3306,
                user='admin',
                password='Dtt@20240625',
                database='ai_algorithm_platform',
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            table_name = "aap_ai_event"

            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
                    if not cursor.fetchone():
                        print(f"Table '{table_name}' does not exist in the database.")
                        return False

                    columns = ', '.join(event_data.keys())
                    placeholders = ', '.join(['%s'] * len(event_data))
                    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

                    cursor.execute(insert_query, tuple(event_data.values()))
                    connection.commit()
                    print(f"{cursor.rowcount} record inserted successfully.")
                    return True

        except Error as e:
            print("Error while connecting to MySQL", e)
            return False

        finally:
            if 'connection' in locals() and connection.open:
                connection.close()
                print("MySQL connection is closed.")




