from datetime import datetime
import os
from torchvision.datasets import ImageFolder

class CusImageFolder(ImageFolder):
    def __getitem__(self, index):
        return super(CusImageFolder, self).__getitem__(index), self.imgs[index][0] # return image path


def save_config(args):
    output_dir = args.out_dir

    log_pth = os.path.join(output_dir, 'config.txt')
    os.system('touch ' + log_pth)

    string = str(datetime.now()) + '\n'
    for key in args.__dict__.keys():
        string += key + ': ' + str(getattr(args, key)) + '\n'

    log_file = open(log_pth, 'a')
    log_file.write(string)
    log_file.close()


def save_grad(args):
    output_dir = args.out_dir

    log_pth = os.path.join(output_dir, 'grad.txt')
    os.system('touch ' + log_pth)

    string = str(datetime.now()) + '\n'
    for key in args.__dict__.keys():
        string += key + ': ' + str(getattr(args, key)) + '\n'

    log_file = open(log_pth, 'a')
    log_file.write(string)
    log_file.close()
