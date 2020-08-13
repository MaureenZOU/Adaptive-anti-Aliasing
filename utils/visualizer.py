from utils import *
import cv2
import os

class Visualizer(object):

    def __init__(self, output_folder, col=21, size=(200,200), demo_name='demo.html'):
        self.out_folder = output_folder
        self.html_file = os.path.join(self.out_folder, demo_name)
        self.col = col
        self.size = size # (h, w)
        self.dir_lst, self.caption_lst = [], []

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
    
    def insert(self, visual_pth, caption, img_pth=None, img=None, save_img=True):
        if save_img == True and img is not None:
            cv2.imwrite(img_pth, img)            
        self.dir_lst.append(visual_pth)
        self.caption_lst.append(caption)

    def write(self, _sorted=True):
        if _sorted == True:
            futils.writeSeqHTML(self.html_file, sorted(self.dir_lst), sorted(self.caption_lst), self.col, self.size[0], self.size[1])
        else:
            futils.writeSeqHTML(self.html_file, self.dir_lst, self.caption_lst, self.col, self.size[0], self.size[1])
