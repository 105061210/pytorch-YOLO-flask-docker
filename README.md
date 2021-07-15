# pytorch-YOLO-flask-docker
##### The model of this project is based on: https://github.com/Tianxiaomo/pytorch-YOLOv4
##### and the flask+docker part is based on the series of tutorial: https://daniel820710.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E5%BE%9E%E9%9B%B6%E5%88%B0%E4%B8%80-day1-%E5%BB%BA%E7%AB%8B-gcp-%E6%A9%9F%E5%99%A8-vm-%E5%9F%B7%E8%A1%8C%E5%80%8B%E9%AB%94-f9efe32377e7
##### For my own detailed implementation, please visit: https://hackmd.io/@sWWbma3eSOOsFTf7i0ax8w/H1f6pyN6O


## 2021/07
## Create VM on GCP
ref: https://daniel820710.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E5%BE%9E%E9%9B%B6%E5%88%B0%E4%B8%80-day1-%E5%BB%BA%E7%AB%8B-gcp-%E6%A9%9F%E5%99%A8-vm-%E5%9F%B7%E8%A1%8C%E5%80%8B%E9%AB%94-f9efe32377e7

### 0. login to Google Cloud Platform
There will be free quota for $US 300

### 1. upgrade your account and add billing information
They will charge you only if you run out of your free quota


### 2. create new project add Compute Engine API to your account
for using VM instances

### 3. modify the GPU quota to the number you want
![](https://i.imgur.com/lcahB9a.png)
<br>
![](https://i.imgur.com/jvHxvo4.png)
They will send you an email once they confirm your apply.
(They said it might take 2 work days, but I got the reply almost in 10 minutes)

### 4. create VM instances
* configure <br>
![](https://i.imgur.com/6LrvQEx.png)
<br>

* operating system <br>
![](https://i.imgur.com/sYACRZn.png)
<br>

* add SSH keys <br>
![](https://i.imgur.com/1SZjHSg.png)
<br>paste your public key
<br>

* then it will look like...<br>
![](https://i.imgur.com/XPIPiTR.png)


---


## Install CUDA 9.0 + cuDNN 7.0.5 on Ubuntu16.04
ref: https://medium.com/cs-note/ubuntu16-04-install-cuda-9-0-cudnn-7-0-5-80c53404516c

### 0. Noveau drivers
* create a new file
```
sudo vi /etc/modprobe.d/blacklist-nouveau.conf
```
* paste the following to the file
```
blacklist nouveau
options nouveau modeset=0
```
* update
```
sudo update-initramfs –u
```
* reboot
```
sudo reboot
```
* after reboot the following command should show nothing
```
lsmod | grep nouveau
```

### 1. CUDA
* update apt-get
```
sudo apt-get update
```
* install apt-get packages
```
sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev curl
```
* install nvidia drivers
```
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
```
```
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
```
```
sudo dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
```
```
sudo apt-get update
```
```
sudo apt-get install cuda-9-0
```
* reboot
```
sudo reboot
```
* check
```
nvidia-smi
```
![](https://i.imgur.com/52qrwgk.png)


### 2. cuDNN
* install cuDNN
```
wget https://s3.amazonaws.com/open-source-william-falcon/cudnn-9.0-linux-x64-v7.1.tgz
```
```
sudo tar -xzvf cudnn-9.0-linux-x64-v7.1.tgz
```
```
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
```
```
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
```
```
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```
* open bashrc
```
sudo vi ~/.bashrc
```
* put the following lines at the end of the file
```bash=
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin
```
* reload
```
source ~/.bashrc
```
---

## Install conda
### 1. conda
```
curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
```
```
bash Anaconda3-5.2.0-Linux-x86_64.sh
```
```
source ~/.bashrc
```
### 2. create environment
```
conda create --name test python=3.6
```
```
source activate
```
### 3. install other packages
* in your new environment
```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
```
(test)$ python
Python 3.6.8 |Anaconda, Inc.| (default, Dec 30 2018, 01:22:34)
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print(torch.cuda.get_device_name(0))
Tesla T4
```
* jupyter notebook
```
conda install -c anaconda jupyter
```
```
conda install ipykernel
```
```
python -m ipykernel install --user --name test(環境名稱) --display-name "test(顯示名稱)"
```
* open a new terminal and ssh to
```
ssh -L 8158:localhost:8888  -i .ssh/id_rsa_gcp lucileee@34.79.19.65
```
```
jupyter notebook --no-browser --port=8888
```
* enter `localhost:8158` in local browser and enter the token shown on remote terminal
![](https://i.imgur.com/ZB2FuG8.png)

### $$ VScode and remote SSH for local editing
* Download vscode
* go to extension and find remote ssh
![](https://i.imgur.com/BlNYcHc.png)

* edit config file in .ssh/config (`/Users/yangyating/.ssh`)
```typescript=
Host 34.79.19.65
  HostName 34.79.19.65
  IdentityFile /Users/yangyating/.ssh/id_rsa_gcp
  User lucileee
```
* go to remote explorer and then connect to the host in current window
* we can access to the remote folder
![](https://i.imgur.com/jygcEl8.jpg)



---

### 4. YOLOv4

#### pytorch

```
git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git
```
example usage:
```
python demo.py -cfgfile cfg/yolov4.cfg -weightfile backup/yolov4.weights -imgfile data/dog.jpg
```

### 5. with Flask
#### modify demo.py in pytorch-YOLOv4 as demo_server.py
```python=
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


###### Add this ######

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

###### Add this ######

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

###### Modify this ######

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

###### Modify this ######

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

```

#### open jupyter notebook and send request

```python=
import requests
import json
from io import BytesIO
from PIL import Image
import base64
```

```python=
req = requests.post('http://127.0.0.1:8891/detect_image', json = {'image':'https://pgw.udn.com.tw/gw/photo.php?u=https://uc.udn.com.tw/photo/2018/03/18/1/4578788.jpg&x=0&y=0&sw=0&sh=0&sl=W&fw=760'})
print(json.loads(req.content)['success'])
```

```python=
image = Image.open(BytesIO(base64.b64decode(json.loads(req.content)['predictions'])))
```

![](https://i.imgur.com/dZjtjyZ.jpg)


### $$ Using API from outside
#### edit VPC in GCP
![](https://i.imgur.com/VTBFz2v.png)
![](https://i.imgur.com/AJCPJgB.png)


#### edit VM instances
![](https://i.imgur.com/c86OAcf.png)

#### result
![](https://i.imgur.com/xxWGHW9.jpg)

#### remote server will look like
![](https://i.imgur.com/uGJbHeP.png)



### 6. Docker

#### install Docker
```
sudo apt-get install docker.io
```
```
service docker status
```
```
sudo usermod -aG docker lucileee
```
```
docker run hello-world
```
#### common docker command
ref: https://blog.gtwang.org/linux/docker-commands-and-container-management-tutorial/
* build image
```
sudo docker build -t mytestimage .
```
* list image
```
docker image ls
```
* remove image
```
docker rmi <image id>
```
* run docker container
```
sudo docker run --runtime=nvidia --gpu 0 -d -p 8891:8891 mytestimage
```
-d: run in background
* list active container (for finding container id)
```
docker container ls
```
* remove inactive container
```
docker container prune
```
* start/stop container
```
docker start <container id>
```
```
docker stop <container id>
```
* command line in container
```
docker exec -it <container id> /bin/bash
```

#### the docker file I use
```dockerfile=
FROM  floydhub/pytorch:1.4.0-gpu.cuda10cudnn7-py3.54

RUN pip install Pillow
RUN pip install flask
RUN pip install requests
RUN pip install scikit_image
RUN pip install matplotlib
RUN pip install tqdm
RUN pip install easydict
RUN pip install numpy
RUN pip install opencv_python==4.0.0.21

WORKDIR /test_yolo
COPY ./pytorch-YOLOv4 ./



EXPOSE 8891
ENTRYPOINT ["bash","demo_run.sh"]
```
#### since https://github.com/Tianxiaomo/pytorch-YOLOv4 work well with
![](https://i.imgur.com/SK25aMr.png)

！！！這裡有一個bug我處理很久，原本是用floydhub/pytorch上最新版本的docker image，但一直報錯 `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` 所以我是根據 https://github.com/werner-duvaud/muzero-general/issues/139
也遇到類似問題的解法去降版本才好的

#### so we build image --> run docker container --> send request
```
sudo docker build -t mytestimage .
sudo docker run --runtime=nvidia --gpu 0 -d -p 8891:8891 mytestimage
```
![](https://i.imgur.com/OpvjwJh.jpg)
#### in order to use --runtime=nvidia we need to install NVIDIA Container Toolkit

ref: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#getting-started

```
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```
```
sudo systemctl restart docker
```
```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
有成功跑出`nvidia-smi`通常會出現的畫面即可

#### push docker image
ref: https://ithelp.ithome.com.tw/articles/10192824


### $$ check version
#### for pytorch and cudnn
![](https://i.imgur.com/KQwJ5Wt.png)

#### for CUDA
```
nvcc --version
```

### $$ increase SSD on GCP

#### Go to compute engine/disk, choose the machine you want to change
![](https://i.imgur.com/LiUVKCC.png)

#### change from 20GB to 40GB (or whatever you want)

#### ssh to your VM

```
sudo lsblk
```
![](https://i.imgur.com/PSK4Tn5.png)

```
sudo growpart /dev/sda 1
```

```
sudo resize2fs /dev/sda1
```
![](https://i.imgur.com/cDlgn3B.png)

