import base64
import datetime
import pickle

import cv2
import numpy as np
import torch
# import torchvision

from flask import Flask, Response
from kafka import KafkaConsumer, KafkaProducer
from PIL import Image
# from detector.detector import get_transform, get_fasterrcnn_model
from data import VOC_CLASSES as labels


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd


def from_base64(buf):
    buf_decode = base64.b64decode(buf)
    buf_arr = np.fromstring(buf_decode, dtype=np.uint8)
    return cv2.imdecode(buf_arr, cv2.IMREAD_UNCHANGED)


topic_in = "raw-video"
topic_out = "object-detections"

net = build_ssd('test', 300, 21)    # initialize SSD
net.load_weights('ssd300_mAP_77.43_v2.pth')


def main():

    # #model = get_fasterrcnn_model()
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model.eval()

    consumer = KafkaConsumer(
        topic_in,
        bootstrap_servers=['kafka1:19091']
    )

    producer = KafkaProducer(bootstrap_servers=['kafka1:19091'])

    image_id = -1

    for msg in consumer:

        #img = decode(msg.value)

        image_id += 1

        img = from_base64(msg.value)
        image = Image.fromarray(img)
        image = np.array(image)

        image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)

        # img = Image.open('./data/sequence-1/img1/000608.jpg').convert("RGB")

        # image_tensor = transform(image, {})
        # image_tensor = image_tensor[0].unsqueeze(0).float()

        # img = cv2.imread(img_path)
        height, width, channels = image.shape
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#cv2.IMREAD_COLOR) #cv2.COLOR_BGR2RGB)

        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)

        xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = net(xx)

        top_k=10

        frame_results_list = []

        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor(image.shape[1::-1]).repeat(2)
        for i in range(detections.size(1)):
            j = 0
            if detections[0,i,j,0] >= 0.6:
                score = detections[0,i,j,0]
                label_name = labels[i-1]
                display_txt = '%s: %.2f'%(label_name, score)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                coords = pt[0], pt[1], pt[2]-pt[0]+1, pt[3]-pt[1]+1

                frame_results_list.append([image_id] + [i] + list(coords) + [score] + [-1, -1, -1])

        frame_results_array = np.array(frame_results_list)

        buffer = pickle.dumps({'image_id':image_id , 'img':msg.value, 'frame_results':frame_results_array})
        producer.send(topic_out, base64.b64encode(buffer))


def old_main():

    #model = get_fasterrcnn_model()
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model.eval()
    topic = "raw-video"
    transform = get_transform(False)

    consumer = KafkaConsumer(
        topic_in,
        bootstrap_servers=['kafka1:19091']
    )

    producer = KafkaProducer(bootstrap_servers=['kafka1:19091'])

    image_id = -1

    for msg in consumer:

        #img = decode(msg.value)

        img = from_base64(msg.value)
        image = Image.fromarray(img)
        image = np.array(image)

        image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)

        # img = Image.open('./data/sequence-1/img1/000608.jpg').convert("RGB")

        image_tensor = transform(image, {})
        image_tensor = image_tensor[0].unsqueeze(0).float()

        with torch.no_grad():
            predictions = model(image_tensor)

        for frame in zip(predictions):

            frame = frame[0]
            image_id += 1
            boxes = frame['boxes'].cpu().numpy()
            labels = frame['labels'].cpu().numpy().reshape(-1,1)
            scores = frame['scores'].cpu().numpy().reshape(-1,1)
            image_id_col = np.array([image_id]*scores.shape[0]).reshape(-1,1)
            x = np.array([-1]*scores.shape[0]).reshape(-1,1)
            y = np.array([-1]*scores.shape[0]).reshape(-1,1)
            z = np.array([-1]*scores.shape[0]).reshape(-1,1)

            # Convert bbox coords to tlwh from tlbr
            boxes[:,2] = boxes[:,2]-boxes[:,0]
            boxes[:,3] = boxes[:,3]-boxes[:,1]

            frame_results = np.hstack([image_id_col, labels, boxes, scores, x, y, z])

        # Convert to bytes and send to kafka
        # producer.send(topic, buffer.tobytes())
        buffer = pickle.dumps({'image_id':image_id , 'img':msg.value, 'frame_results':frame_results})
        producer.send(topic_out, base64.b64encode(buffer))


if __name__ == '__main__':
    main()
