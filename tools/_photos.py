"""Load the reference photograph of a captured view, for score_views / compose_views / compare_trainer_viewer.

A dataset photo is fetched from the app by URL. A VIDEO frame (photo 'video-frame:<name>') only ever existed in
the app's memory, so the harness saves it next to the capture as <Dataset>__<tag>__view-<k>-photo.png
(Studio.StashVideoPhotoAsync) and it is read from there.
"""
import io
import os
import urllib.request

from PIL import Image

SHOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shots', 'dataset')


def load_photo(side, name, tag, view_key):
    photo = side['views'][view_key]['photo']
    if photo.startswith('video-frame:'):
        return Image.open(os.path.join(SHOTS, f'{name}__{tag}__view-{view_key}-photo.png')).convert('RGB')
    with urllib.request.urlopen(side['app'].rstrip('/') + '/' + photo.lstrip('/')) as r:
        return Image.open(io.BytesIO(r.read())).convert('RGB')
