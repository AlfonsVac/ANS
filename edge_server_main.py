import argparse
import cv2
import torchvision.transforms as transforms
import torch

from PIL import Image

from models.vgg16 import vgg16
from models.tiny_yolo import tinyYolo
from models.vc.vc_torch import vc
from communication import serverCommunication


WINDOW_NAME = 'CameraDemo'
device = torch.device("cpu")

def parse_args():
    # Parse input arguments
    desc = 'ANS in edge server side'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dnn', dest='dnn_model',
                        help='vgg, yolo, vc',
                        default='yolo', type=str)
    parser.add_argument('--host', dest='host',
                        help='Ip address',
                        default='192.168.1.72', type=str)
    parser.add_argument('--port', dest='port',
                        help='Ip port',
                        default=8080, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))

    if args.dnn_model == 'vgg':
        model = vgg16()
        model.eval()
    elif args.dnn_model == 'vc':
        model = vc()
        model.eval()
    else:
        model = tinyYolo()
        model.eval()

    model.to(device)

    communication = serverCommunication(args.host, args.port)

    while True:
        try:
            conn, addr = communication.accept_conn()
            with conn:
                recv_data = communication.receive_msg(conn)
                print('receive data from mobile device !!!')
                partition_point = recv_data[0]
                data = recv_data[1]
                data = torch.autograd.Variable(data)
                prediction = model(data.to(device), server=True, partition=partition_point)
                res = prediction.data

                msg = communication.send_msg(conn, res)

        except KeyboardInterrupt or TypeError or OSError:
            communication.close_channel()
