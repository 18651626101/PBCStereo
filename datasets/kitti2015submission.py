import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
import torchvision
from PIL import ImageFilter


class KITTI2015submission(Dataset):

    def __init__(self, datapath, list_filename):
        self.datapath = datapath
        self.left_filenames, self.right_filenames = self.load_path(list_filename)

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images

    def load_image(self, filename):
        return Image.open(filename).convert('L')

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        w, h = left_img.size
        top_pad = 384 - h
        right_pad = 1280 - w
        assert top_pad > 0 and right_pad > 0

        left_img = np.ascontiguousarray(left_img, dtype=np.float32)
        right_img = np.ascontiguousarray(right_img, dtype=np.float32)

        left_img = np.lib.pad(
            left_img,
            ((top_pad, 0), (0, right_pad)),
            mode='symmetric',
        )
        right_img = np.lib.pad(right_img, ((top_pad, 0), (0, right_pad)), mode='symmetric')

        preprocess = get_transform()
        left_img = preprocess(left_img)
        right_img = preprocess(right_img)

        # return [left_img,right_img],-disparity
        return {"left": left_img, "right": right_img}
