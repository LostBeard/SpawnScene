"""Build datasets/TruckVideo/truck.mp4 from the Truck dataset images: one image per frame at 10 fps, H.264.

    pip install imageio-ffmpeg
    python tools/make_truck_video.py

Frame k of the video is Truck image k (the manifest's order), so TruckVideo/poses.par - Truck's COLMAP poses renamed
to frame_0001.jpg ... - is ground truth for the video path's pose-vs-GT report. Default x264 GOP (one keyframe for
the whole 12.6 s clip) on purpose: the worst case for seeking.
"""
import json
import os
import subprocess

import imageio_ffmpeg

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
man = json.load(open(os.path.join(ROOT, 'SpawnScene', 'wwwroot', 'datasets', 'Truck', 'manifest.json')))
src = os.path.join(man['source'], man['imageDir'])
lst = os.path.join(ROOT, '_scratch', 'truck_frames.txt')
os.makedirs(os.path.dirname(lst), exist_ok=True)
with open(lst, 'w') as f:
    for n in man['images']:
        f.write(f"file '{os.path.join(src, n).replace(os.sep, '/')}'\nduration 0.1\n")
    f.write(f"file '{os.path.join(src, man['images'][-1]).replace(os.sep, '/')}'\n")
out = os.path.join(ROOT, 'SpawnScene', 'wwwroot', 'datasets', 'TruckVideo', 'truck.mp4')
os.makedirs(os.path.dirname(out), exist_ok=True)
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-f', 'concat', '-safe', '0', '-i', lst,
                '-vf', 'fps=10,scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '18', '-movflags', '+faststart', out], check=True)

# Truck's poses, renamed frame by frame.
poses = open(os.path.join(ROOT, 'SpawnScene', 'wwwroot', 'datasets', 'Truck', 'poses.par')).read().splitlines()
n = int(poses[0])
lines = [str(n)] + [f'frame_{k + 1:04d}.jpg {line.split(" ", 1)[1]}' for k, line in enumerate(poses[1:1 + n])]
open(os.path.join(os.path.dirname(out), 'poses.par'), 'w', newline='\n').write('\n'.join(lines) + '\n')
print(out, os.path.getsize(out), 'bytes')
