from flask import Flask, request, jsonify
from multiprocessing import Process ,Queue
from utils.Post_data import Get_post_data
from main import run

app = Flask(__name__)
processes = {}
out_processes = {}
id_list = []


def add_id():
    global id_list
    if not id_list:
        new_id = 0
    else:
        new_id = id_list[-1] + 1
    id_list.append(new_id)
    return new_id


def start_detect(id, data, rtsp_url):
    pw = Process(target=run, args=(id, data))
    pw.start()
    processes[rtsp_url] = pw
    print(f"Starting detection for {rtsp_url}")


@app.route('/dataSynch/ai/pushCameraDetectionParams', methods=['POST'])
def main():
    data = request.get_json()
    print(data)

    # if not Get_post_data.check(data[0]['rtspUrl']):
    #     return jsonify({'message': 'RTSP URL check failed', 'code': 400}), 400

    res = Get_post_data.check_all_data(data)
    if res:
        return jsonify({'message': res, 'code': 400}), 400

    if data[0]['rtspUrl'] in processes:
        old_process = processes.pop(data[0]['rtspUrl'], None)
        old_process.terminate()
        old_process.join()

        old_process_1 = out_processes.pop(data[0]['rtspUrl'], None)
        old_process_1.terminate()
        old_process_1.join()
        print(f"Restarted previous process for {data[0]['rtspUrl']}")

    id = add_id()
    start_detect(id, data, data[0]['rtspUrl'])
    return jsonify({'message': 'success', 'code': 200}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009, use_reloader=False)



