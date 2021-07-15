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
