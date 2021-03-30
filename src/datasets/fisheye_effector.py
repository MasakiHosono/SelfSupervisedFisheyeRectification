import argparse
import io
import os
import sys
import numpy as np
from scipy.sparse import lil_matrix, find
from PIL import Image
from math import sqrt

import torch

import time

USE_SPARSE = True
DEVICE = 'cpu'

class FisheyeEffector:
    def __init__(self, height=720, width=1280, distortion=0.5):
        self.float_height, self.float_width = float(height), float(width)
        self.height, self.width = height, width
        self.setDistortion(distortion=distortion)

    def setDistortion(self, distortion=0.5):
        self.distortion = distortion
        self.crop = distortion > 0
        self.left, self.upper, self.right, self.lower = 0, 0, self.width, self.height
        self.key_coordinates = []

        # calculate filter
        num_pixels = self.width * self.height
        i_idx = []
        j_idx = []
        for i in range(num_pixels):
            h = int(i / self.width)
            w = int(i % self.width)

            norm_h, norm_w = (2*h - self.height) / self.height, (2*w - self.width) / self.width
            diagonal = norm_h - norm_w == 0
            norm_h = norm_h * self.float_height / self.float_width

            radius = sqrt(norm_h**2 + norm_w**2)
            org_norm_h, org_norm_w = calc_points_of_original_image(norm_h, norm_w, radius, distortion)

            org_norm_h = org_norm_h * self.float_width / self.float_height
            org_h, org_w = int((org_norm_h * self.height + self.height) / 2), int((org_norm_w * self.width + self.width) / 2)

            j = org_h * self.width + org_w
            if j not in range(num_pixels):
                continue

            i_idx.extend(range(3*i, 3*i + 3))
            j_idx.extend(range(3*j, 3*j + 3))

            # remember coordinates for cropping result images
            if diagonal:
                if self.left == 0 and self.upper == 0:
                    self.left, self.upper = w, h
                self.right, self.lower = w, h

        i = torch.tensor([i_idx, j_idx])
        v = torch.tensor([1.0 for _ in range(len(i_idx))])
        self.filter = torch.sparse_coo_tensor(i, v, (num_pixels * 3, num_pixels * 3)).to(DEVICE)

        # calculate key coordinates
        candidate_norms = [(0.2, 0.1), (0.2, 0), (0.2, -0.1)]
        expansion = 1.0
        if self.crop:
            expansion = self.float_width / float(self.right - self.left)
        for norm_x, norm_y in candidate_norms:
            radius = sqrt(norm_x**2 + norm_y**2)
            dst_x, dst_y = calc_points_of_distorted_image(norm_x, norm_y, radius, distortion)
            self.key_coordinates.append((dst_x * expansion, dst_y * expansion))

        print('FisheyeEffector was initialized with distortion = {}'.format(distortion))

    def getKeyCoordinates(self):
        return self.key_coordinates

    def getDistortion(self):
        return self.distortion

    def apply(self, image_bytes):
        return self.calcImage(Image.open(io.BytesIO(image_bytes)))

    def __call__(self, image):
        return Image.open(
            io.BytesIO(
                self.calcImage(image)
            )
        )

    def calcImage(self, image):
        image = np.array(image)

        # padding
        image = padding(image, height=self.height, width=self.width)

        org_dtype = image.dtype
        org_shape = image.shape

        # apply filter
        with torch.no_grad():
            image = torch.from_numpy(image.reshape(-1, 1)).float().to(DEVICE)
            fish_image = torch.sparse.mm(self.filter, image)
            fish_image = fish_image.to('cpu').detach().numpy().astype(org_dtype).reshape(org_shape)

        fish_image = Image.fromarray(fish_image)
        if self.crop:
            fish_image = fish_image.crop((self.left, self.upper, self.right, self.lower))
        fish_image = fish_image.resize((self.width, self.height), Image.LANCZOS)

        fish_image_bytes = io.BytesIO()
        fish_image.save(fish_image_bytes, 'png')

        return fish_image_bytes.getvalue()

def calc_points_of_original_image(x, y, r, distortion):
    if distortion > 1:
        distortion = 1
    elif distortion < -1:
        distortion = -1

    if 1 - distortion*(r**2) == 0:
        return x, y

    return x / (1 - distortion*(r**2)), y / (1 - distortion*(r**2))

def calc_points_of_distorted_image(x, y, r, distortion):
    if distortion > 1:
        distortion = 1
    elif distortion < -1:
        distortion = -1

    if distortion == 0 or x == 0:
        return x, y

    a = distortion * x * (1 + y**2/x**2)
    c = -x

    new_x = (-1 + sqrt(1 - 4*a*c)) / (2*a)
    new_y = (y/x) * new_x

    return new_x, new_y

def padding(image, height=720, width=1280):
    src_height, src_width, _ = image.shape
    if src_height < height and src_width < width:
        pad_h = int((height-src_height)/2)
        pad_w = int((width-src_width)/2)
        image = np.pad(image, [(pad_h, pad_h), (pad_w, pad_w), (0, 0)], 'constant')
    return image

def execDir(effector, path):
    for file in os.listdir(path):
        joined_path = os.path.join(path, file)

        if os.path.isfile(joined_path):
            _, ext = os.path.splitext(joined_path)
            output_path = joined_path.replace(ext, '_dis.png')
            execFile(effector, joined_path, output_path=output_path)

        else:
            execDir(effector, joined_path)

def execFile(effector, input_path, output_path='output.png'):
    if input_path.endswith('_dis.png'):
        print('skip', input_path)
        return

    if os.path.exists(output_path):
        print('skip', input_path)
        return

    with open(input_path, 'rb') as image_bin:
        print(input_path)
        output = open(output_path, 'wb')
        output.write(effector.apply(image_bin.read()))
        output.close()

def resizeAndApply(effector, width, height, input_path, output_path='output.png'):
    print(input_path)
    image = Image.open(input_path)
    image = image.resize((width, height))
    image = effector(image)
    image.save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help='path to the input image or directory')
    parser.add_argument('-d', '--distortion', type=float, default=0.1, help='amount of distortion between -1 to 1 (0.1 as default)')
    parser.add_argument('--width', type=int, default=1280, help='input image width (1280 as default)')
    parser.add_argument('--height', type=int, default=720, help='input image height (720 as default)')

    args       = parser.parse_args()
    input_path = args.input
    distortion = args.distortion
    width      = args.width
    height     = args.height

    if not os.path.exists(input_path):
        print('No such file or directory: {}'.format(input_path))
        exit(1)

    start = time.time()
    effector = FisheyeEffector(height=height, width=width, distortion=distortion, backward=False)
    end = time.time()

    print('{} sec had been spent for Initialize.'.format(end-start))

    if os.path.isfile(input_path):
        start = time.time()
        resizeAndApply(effector, width, height, input_path)
        end = time.time()

    else:
        execDir(effector, input_path)
