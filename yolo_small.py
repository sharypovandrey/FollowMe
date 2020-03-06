# https://towardsdatascience.com/pedestrian-tracking-in-real-time-using-yolov3-33439125efdf
import os
from os.path import exists, join, basename

project_name = "deep_sort_pytorch"
if not exists(project_name):
  # clone and install
  !git clone -q --recursive https://github.com/ZQPei/deep_sort_pytorch.git
  
import sys
sys.path.append(project_name)
sys.path.append(join(project_name, 'YOLOv3'))

import IPython
from IPython.display import clear_output

if not exists('yolov3.weights'):
  !wget -q https://drive.google.com/file/d/1I92QB0ifKoEOZcUq6gAfay-o4Tjw1xTU/view?usp=sharing

import cv2
import time

from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from util import draw_bboxes

yolo3 = YOLOv3("deep_sort_pytorch/YOLOv3/cfg/yolo_v3.cfg","yolov3.weights","deep_sort_pytorch/YOLOv3/cfg/coco.names", is_xywh=True)
deepsort = DeepSort("ckpt.t7")


VIDEO_URL = 'http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentreXVID.avi'
DURATION_S = 20  # process only the first 20 seconds

video_file_name = 'video.mp4'
if not exists(video_file_name):
  !wget -q $VIDEO_URL
  dowloaded_file_name = basename(VIDEO_URL)
  # convert to MP4, because we can show only MP4 videos in the colab noteook
  !ffmpeg -y -loglevel info -t $DURATION_S -i $dowloaded_file_name $video_file_name
  
def show_local_mp4_video(file_name, width=640, height=480):
  import io
  import base64
  from IPython.display import HTML
  video_encoded = base64.b64encode(io.open(file_name, 'rb').read())
  return HTML(data='''<video width="{0}" height="{1}" alt="test" controls>
                        <source src="data:video/mp4;base64,{2}" type="video/mp4" />
                      </video>'''.format(width, height, video_encoded.decode('ascii')))
 
clear_output()
video = show_local_mp4_video('video.mp4')
video



video_capture = cv2.VideoCapture()
if video_capture.open('video.mp4'):
  width, height = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = video_capture.get(cv2.CAP_PROP_FPS)
  !rm -f output.mp4 output.avi
  # can't write out mp4, so try to write into an AVI file
  video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
  while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
      break
      
    start = time.time()
    xmin, ymin, xmax, ymax = 0, 0, width, height
    im = frame[ymin:ymax, xmin:xmax, (2,1,0)]
    bbox_xywh, cls_conf, cls_ids = yolo3(im)
    if bbox_xywh is not None:
        mask = cls_ids==0
        bbox_xywh = bbox_xywh[mask]
        bbox_xywh[:,3] *= 1.2
        cls_conf = cls_conf[mask]
        outputs = deepsort.update(bbox_xywh, cls_conf, im)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:,:4]
            identities = outputs[:,-1]
            frame = draw_bboxes(frame, bbox_xyxy, identities, offset=(xmin,ymin))

    end = time.time()
    print("time: {}s, fps: {}".format(end-start, 1/(end-start)))
            
    video_writer.write(frame)
  video_capture.release()
  video_writer.release()
  
  # convert AVI to MP4
  !ffmpeg -y -loglevel info -i output.avi output.mp4
else:
  print("can't open the given input video file!")


show_local_mp4_video('output.mp4', width=960, height=720)
