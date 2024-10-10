import cv2
import numpy as np
from shapely.geometry import Polygon

class Get_post_data:
    def __init__(self):
        self.default_detection_info = {
                                        'detectionType': '',
                                        'detectionAreas': [],
                                        'isEnabled': "False",
                                        'activeTime': ['00:00:01', '23:59:59'],
                                        'alarmInhibitionTime': 0,
                                        'triggerTime': 0,
                                        'recordTime': 0}

    def data_feng(self,data_list, w, h):
        det1 = self.default_detection_info.copy()
        det2 = self.default_detection_info.copy()
        det3 = self.default_detection_info.copy()
        det4 = self.default_detection_info.copy()
        det5 = self.default_detection_info.copy()
        det6 = self.default_detection_info.copy()
        for data in data_list:
            info = self.get_info(data, w, h)
            if info['detectionType'] == '1':
                det1 = info
            elif info['detectionType'] == '2':
                det2 = info
            elif info['detectionType'] == '3':
                det3 = info
            elif info['detectionType'] == '4':
                det4 = info
            elif info['detectionType'] == '5':
                det5 = info
            elif info['detectionType'] == '6':
                det6 = info
        return det1, det2, det3, det4, det5, det6

    def get_info(self,data, w, h):
        direction_data = self.get_points(data, w, h)  # 调用get_points函数，直接使用w和h
        detection_type = data['detectionType']
        detection_areas = []

        for area in data.get('detectionAreas', []):
            # 直接使用w和h作为resolution
            resolution = [float(w), float(h)]
            detection_areas.append({'anchors': area['anchors'], 'resolution': resolution})

        if detection_areas:
            detection_areas = self.convert_anchors_to_pixels(detection_areas)
            pts, polygons = self.jisuan_pts(detection_areas)
        else:
            detection_areas = []
            pts = []
            polygons = []

        result = {
            'cameraNum': data['cameraNum'],
            'cameraName': data['cameraName'],
            'rtspUrl': data['rtspUrl'],
            'detectionClassify': data['detectionClassify'],
            'detectionType': detection_type,
            'detectionAreas': detection_areas,
            'pts': pts,
            'polygons': polygons,
            'direction_datas': direction_data,
            'isEnabled': data['isEnabled'],
            'activeTime': data['activeTime'],
            'alarmInhibitionTime': data['alarmInhibitionTime'],
            'triggerTime': data['triggerTime'],
            'recordTime': data['recordTime'],
            'triggerTargetNum': data.get('triggerTargetNum', 15),
            'confidenceThreshold': data.get('confidenceThreshold', 0.5),
            'overlapRate': data.get('overlapRate', 0.5),
            'alarmLevel': data.get('alarmLevel', 2)
        }
        return result

    def get_points(self,data, w, h):
        data = [data]
        all_points = []  # 这个列表将用来收集所有的字典

        for camera in data:
            for area in camera.get('detectionAreas', []):
                result_dict = {}  # 在这里初始化，每个 area 一个新的字典

                point_data = area.get('point', [])
                if len(point_data) > 0:
                    resolution = (w, h)
                    points = {'anchors': area.get('anchors', []), 'resolution': resolution}
                    detection_areas = self.convert_anchors_to_pixels([points])
                    _, polygons = self.jisuan_pts(detection_areas)
                    result_dict["polygons"] = polygons  # 添加多边形数据

                    for points in point_data:
                        point1 = (points[0], points[1])
                        point2 = (points[2], points[3])
                        point_data1 = self.calculate_vector_direction(point1, point2)
                        result_dict["vector_direction"] = point_data1

                    all_points.append(result_dict)
        return all_points

    @staticmethod
    def check_data_entry(data):
        required_fields = {
            'cameraNum': {'type': str, 'max_length': None},
            'cameraName': {'type': str, 'max_length': None},
            'rtspUrl': {'type': str, 'max_length': None},
            'detectionClassify': {'type': str, 'max_length': 2},
            'detectionType': {'type': str, 'max_length': 2},
            'detectionAreas': {'type': list, 'max_length': None},
            'isEnabled': {'type': str, 'max_length': None},
            'activeTime': {'type': list, 'max_length': None},
            'alarmLevel': {'type': str, 'max_length': 1},
            'alarmType': {'type': list, 'max_length': None},
            'alarmInhibitionTime': {'type': int, 'max_length': None},
            'isInhibitFromStart': {'type': str, 'max_length': 1},
            'triggerTime': {'type': int, 'max_length': None},
            'recordTime': {'type': int, 'max_length': None}
        }

        optional_fields = {
            'staffRecognization': {'type': int, 'max_length': 1},
            'triggerTargetNum': {'type': int, 'max_length': None},
            'sensitivity': {'type': float, 'max_length': None},
            'confidenceThreshold': {'type': float, 'max_length': None},
            'overlapRate': {'type': float, 'max_length': None}
        }

        errors = {}

        for field, specs in {**required_fields, **optional_fields}.items():
            if field in data:
                if not isinstance(data[field], specs['type']):
                    errors[field] = f'Expected type {specs["type"].__name__}, got {type(data[field]).__name__}.'
                elif specs['max_length'] is not None and isinstance(data[field], str) and len(data[field]) > specs[
                    'max_length']:
                    errors[field] = f'Exceeds maximum length of {specs["max_length"]}.'

        if 'detectionAreas' in data:
            for area in data['detectionAreas']:
                if not isinstance(area, dict) or 'anchors' not in area or 'resolution' not in area:
                    errors['detectionAreas'] = 'Invalid detectionAreas format.'
                    break

        return errors

    def jisuan_pts(self,data_lists):
        pts_list = []
        polygons = []
        for points in data_lists:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            pts_list.append(pts)
            polygon = Polygon(points)
            polygons.append(polygon)
        return pts_list, polygons

    def convert_anchors_to_pixels(sefl,elements):
        converted_anchors = []  # 用于存储转换后的像素坐标
        for element in elements:
            anchors = element['anchors']  # 锚点列表
            resolution = element['resolution']  # 分辨率
            pixel_anchors = [[int(x * resolution[0]) / 100.0, int(y * resolution[1]) / 100.0] for x, y in anchors]
            converted_anchors.append(pixel_anchors)
        return converted_anchors

    def calculate_vector_direction(sefl,point1, point2):
        # 计算向量 (dx, dy)
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        direction = (dx, dy)
        return direction

    def resize_and_pad(self,w, h, desired_size=640):
        ratio = float(desired_size) / max(w, h)
        new_size = tuple([int(x * ratio) for x in (w, h)])
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        return ratio, (left, top)

    @staticmethod
    def check(rtsp):
        cap = cv2.VideoCapture(rtsp)
        if not cap.isOpened():
            print(f"Unable to open video stream: {rtsp}")
            return False
        cap.release()
        return True

    @staticmethod
    def check_all_data(data_list):
        results = []
        for data in data_list:
            result = Get_post_data.check_data_entry(data)
            if result:
                results.append(result)
        if results:
            return True
        else:
            return False


