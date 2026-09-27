"""Load the reference photograph of a captured view, for score_views / compose_views / compare_trainer_viewer.

A dataset photo is read from disk (wwwroot, or its manifest's source mount), else fetched from the app by URL.
A VIDEO frame (photo 'video-frame:<name>') only ever existed in the app's memory, so the harness saves it next
to the capture as <Dataset>__<tag>__view-<k>-photo.png (Studio.StashVideoPhotoAsync) and it is read from there.
"""
import io
import json
import os
import urllib.request

from PIL import Image

SHOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shots', 'dataset')
WWWROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SpawnScene', 'wwwroot')


def local_photo_path(photo):
    parts = photo.lstrip('/').split('/')
    local = os.path.join(WWWROOT, *parts)
    if os.path.isfile(local):
        return local
    # datasets/<Name>/<imageDir>/<file...>: mounted from the manifest's source directory.
    if len(parts) >= 4 and parts[0] == 'datasets':
        try:
            with open(os.path.join(WWWROOT, 'datasets', parts[1], 'manifest.json'), encoding='utf-8') as f:
                m = json.load(f)
        except (OSError, ValueError):
            return None
        if m.get('source') and m.get('imageDir') == parts[2]:
            local = os.path.join(m['source'], m['imageDir'], *parts[3:])
            if os.path.isfile(local):
                return local
    return None


def load_photo(side, name, tag, view_key):
    photo = side['views'][view_key]['photo']
    if photo.startswith('video-frame:'):
        return Image.open(os.path.join(SHOTS, f'{name}__{tag}__view-{view_key}-photo.png')).convert('RGB')
    # Read the photo from disk so scoring does not need the app running: the same file the server sends,
    # either under wwwroot or, for a large dataset, under its manifest's source mount (tools/_spa_server.js).
    local = local_photo_path(photo)
    if local:
        return Image.open(local).convert('RGB')
    with urllib.request.urlopen(side['app'].rstrip('/') + '/' + photo.lstrip('/')) as r:
        return Image.open(io.BytesIO(r.read())).convert('RGB')
