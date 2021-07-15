# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import requests
from io import BytesIO
from base64 import encodebytes
from PIL import Image
import flask

"""hyper parameters"""
use_cuda = True
app = flask.Flask(__name__)


def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)


def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

    cap.release()


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@app.route("/detect_image", methods=["POST"])
def detect_image():
    output_dict = {"success": False}
    if flask.request.method == "POST":
        data = flask.request.json
        
        # read the image in PIL format
        response = requests.get(data["image"])
        image = Image.open(BytesIO(response.content))
        image.save("temp.jpg")
        # transform image
        #image_tensor = data_transforms(image).float()
        #image_tensor = image_tensor.unsqueeze_(0).to(device)

        # predict and max
        #output = model(image_tensor)
        #_, predicted = torch.max(output.data, 1)
        detect_cv2(args.cfgfile, args.weightfile, "temp.jpg")
        output_dict["predictions"] = get_response_image("predictions.jpg")
        output_dict["success"] = True
    return flask.jsonify(output_dict), 200


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    #parser.add_argument('-imgfile', type=str,
    #                    default='./data/mscoco2017/train2017/190109_180343_00154162.jpg',
    #                    help='path of your image file.', dest='imgfile')
    parser.add_argument('-port', help='port', type=int, default=5000)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    '''
    if args.imgfile:
        detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_imges(args.cfgfile, args.weightfile)
        # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile)
    '''
    app.run(host="0.0.0.0", debug=True, port=args.port)
