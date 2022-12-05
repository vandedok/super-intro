import logging
import os
from tempfile import TemporaryDirectory
from argparse import ArgumentParser
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
from pytube import YouTube 
from superintro.detect import DetectorForFrames
logging.basicConfig()

parser = ArgumentParser(prog = 'YotubeClipperDetector')
parser.add_argument('--link', required=True)
parser.add_argument('--out_dir',required=True)   
parser.add_argument('--duratation', default=50, type=int)       

args = parser.parse_args()

try: 
    yt = YouTube(args.link) 
except: 
    print("Connection Error") 

det_config = '../../mmdetection/configs/fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py'
det_checkpoint = '../../data/mmcv/mmdet/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco-d92ceeea.pth'
detector = DetectorForFrames(det_config, det_checkpoint)

with TemporaryDirectory() as temp_dir:
    
    logging.info("Downloading...")
    video = yt.streams.filter(subtype='mp4', res="480p").first().download(temp_dir)
    os.rename(video, os.path.join(temp_dir, "video.mp4"))
    audio = yt.streams.filter(only_audio=True).first().download(temp_dir)
    os.rename(video, os.path.join(temp_dir,"audio.mp4"))
    video_name = os.path.basename(video)
    video = VideoFileClip(os.path.join(temp_dir, "video.mp4"))
    audio = AudioFileClip(os.path.join(temp_dir, "audio.mp4"))
    logging.info("Done!")

    video = video.set_audio(audio)
    time_start = np.random.random()*(video.duration - args.duratation)
    time_stop = time_start + args.duratation
    out_path = os.path.join(args.out_dir, video_name)
    clip = video.subclip(time_start, time_stop)
    detected = clip.fl_image(detector.frame_with_bboxes)
    detected.write_videofile(out_path)
    
