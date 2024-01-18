# Inspired from https://github.com/athn-nik/sinc

import moviepy.editor as mp
import moviepy.video.fx.all as vfx
import os
import itertools
import subprocess
import numpy as np
from typing import List
from moviepy.editor import VideoFileClip, clips_array, vfx


mpy_conf = {"codec": "libx264",
            "audio_codec": "aac",
            "temp_audiofile": "temp-audio.m4a",
            "remove_temp": True,
            "bitrate": "500k",
            "verbose":False,
            "logger": None,
            "fps": 30}

def stack_vids_moviepy(video_lst, dur, savepath=None):

    if isinstance(video_lst[0], str):
        video_lst = [VideoFileClip(vp) for vp in video_lst]
    if len(video_lst) < 3:
        n = 2
    elif len(video_lst) % 3 == 0:
        n = 3
    elif len(video_lst) % 4 == 0:
        n = 4
    else:
        if len(video_lst) % 2:
            video_lst.append(video_lst[-1])
        n = len(video_lst) // 2

    video_lst = [video_lst[i:i+n] for i in range(0, len(video_lst), n)]
    final_clip = clips_array(video_lst)
    # breakpoint()
    if savepath is not None:
        final_clip.duration = dur
        final_clip.write_videofile(f'{savepath}',  **mpy_conf)
    # needs ImageMagick
    return final_clip


def add_text_moviepy(video, text, position='bottom', fontsize=25):
    if isinstance(video, str):
        video = VideoFileClip(video)
    align = 'center'
    # needs ImageMagick
    video_text = mp.TextClip(text,
                            font='Amiri',
                            color='white',
                            method='caption',
                            align=align,
                            size=(video.w, None),
                            fontsize=fontsize)
    if 'bottom' in position:
        video_text = video_text.on_color(size=(video.w, video_text.h + 5),
                                                color=(0, 0, 0),
                                                col_opacity=0.6)

    else:
        video_text = video_text.on_color(size=(video.w//2, (video_text.h + 30)//2),
                                            color=(0, 0, 0),
                                            col_opacity=0.6)

    if isinstance(position[0], str):
        video_text = video_text.set_pos(position)
    else:
        video_text = video_text.set_pos(position, relative=True)

    video = mp.CompositeVideoClip([video, video_text])
    return video

class Video:
    def __init__(self, frames: str, fps: float = 30) -> None:
        # frame_path = str(frame_path)
        self.fps = fps

        self._conf = {"codec": "libx264",
                      "fps": self.fps,
                      "audio_codec": "aac",
                      "temp_audiofile": "temp-audio.m4a",
                      "remove_temp": True, "verbose":False,
                      "logger":None}

        self._conf = {"bitrate": "5000k",
                      "fps": self.fps,"verbose":False,
                      "logger":None}

        # Load video
        # video = mp.VideoFileClip(video1_path, audio=False)
        # Load with frames

        if isinstance(frames, str) and frames.endswith('.mp4'):
            video = VideoFileClip(frames)
        elif isinstance(frames, str) and frames.endswith('.png'):
            video = VideoFileClip(frames)
        elif isinstance(frames, list):
            video = mp.ImageSequenceClip(frames, fps=fps)
        else:
            frames = [os.path.join(frames, x) for x in sorted(os.listdir(frames))]
            video = mp.ImageSequenceClip(frames, fps=fps)
        self.video = video
        self.duration = video.duration

    def add_text(self, text):
        # needs ImageMagick
        video_text = mp.TextClip(text,
                                 font='Amiri',
                                 color='white',
                                 method='caption',
                                 align="center",
                                 size=(self.video.w, None),
                                 fontsize=30)
        video_text = video_text.on_color(size=(self.video.w, video_text.h + 5),
                                         color=(0, 0, 0),
                                         col_opacity=0.6)
        # video_text = video_text.set_pos('bottom')
        video_text = video_text.set_pos('bottom')

        self.video = mp.CompositeVideoClip([self.video, video_text])

    def save(self, out_path) -> str:
        # import moviepy.editor as mpy
        # from moviepy.video.fx.all import crop
        # (w, h) = self.video.size
        # self.video = crop(self.video, width=900, height=900, x_center=w/2, y_center=h/2)
        out_path = str(out_path)
        self.video.subclip(0, self.duration).write_videofile(out_path, **self._conf)
        return out_path

def stack_vids(vids_to_stack: List[str], fname: str, orient='v', v=False):
    cmd_m = ['ffmpeg']

    vids_arg = list(itertools.chain(*[('-i', j) for j in vids_to_stack]))
    cmd_m.extend(vids_arg)

    cmd_m.extend(['-y', '-loglevel', 'quiet', '-filter_complex',
                    f'{orient}stack=inputs={len(vids_to_stack)}',
                    f'{fname}'])
    if v:
        print('Executing', ' '.join(cmd_m))
    x = subprocess.call(cmd_m)

def put_text(text: str, fname: str, outf: str, v=False):
    cmd_m = ['ffmpeg']
    # -i inputClip.mp4 -vf f"drawtext=text='{method}':x=200:y=0:fontsize=22:fontcolor=white" -c:a copy {temp_path}.mp4

    cmd_m.extend(['-i',fname, '-y', '-vf', f"drawtext=text='{text}':x=(w-text_w)/2:y=h-th-10:fontsize=20::box=1:boxcolor=black@0.6:boxborderw=5:fontcolor=white",
                    '-loglevel', 'quiet', '-c:a', 'copy',
                    f'{outf}'])

    if v:
        print('Executing', ' '.join(cmd_m))
    x = subprocess.call(cmd_m)

    return outf


def save_video_samples(vid_array, video_path, text, fps=30):
    import cv2
    vid_path = f'{video_path}'
    # vid_array is (B, T, 3, W, H)
    w, h = vid_array.shape[-2], vid_array.shape[-1]
    fps = fps
    label = text
    if len(vid_array.shape) > 4:
        duration = vid_array.shape[1] // fps
        no_vids = vid_array.shape[0]
        # (B, T, 3, W, H) -> (T, B, W, H, 3)
        vid_array = np.transpose(vid_array,(1, 0, 3, 4, 2))

    else:
        duration = vid_array.shape[0] // fps
        no_vids = 1

    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (no_vids*w, h)
                          )

    for img in vid_array:
        if len(vid_array.shape) > 4:
            img = np.uint8(np.hstack(img[:]))
            img_ar = img.copy()
            # img_ar = np.uint8(np.transpose(img.copy(), (1, 2, 0)))
        else:
            img_ar = np.uint8(np.transpose(img, (1, 2, 0)))
        for i in range(no_vids+1):
            cv2.putText(img_ar, f'{label}', (w*i + 10, 25),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, cv2.LINE_AA)
        out.write(img_ar)
    out.release()

    # assert retcode == 0, f"Command {' '.join(cmd)} failed, in function write_video_cmd."
    return vid_path

def read_vid(path):
    import cv2
    # Import the video and cut it into frames.
    vid = cv2.VideoCapture(path)
    frames = []
    check = True
    i = 0

    while check:
        check, arr = vid.read()
        frames.append(arr)
        i += 1

    return np.array(frames)  # convert list of frames to numpy array
