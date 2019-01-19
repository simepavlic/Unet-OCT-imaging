from PIL import Image
import os
import shutil


def compare(orig_path, mask_path):
    orig = Image.open(orig_path)
    mask = Image.open(mask_path)
    orig.show()
    mask.show()


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    orig_dir = os.path.join(project_root + '/../input/test')
    mask_dir = os.path.join(project_root + '/../output')
    files = os.listdir(orig_dir)

    for file in files:
        base, ext = os.path.splitext(file)
        orig_path = os.path.join(orig_dir, file)
        mask_path = os.path.join(mask_dir, file)
        dest_path = os.path.join(mask_dir, base)
        dest_orig_path = os.path.join(dest_path, file)
        dest_mask_path = os.path.join(dest_path, base + '_mask' + ext)
        os.makedirs(dest_path)
        shutil.copyfile(orig_path, dest_orig_path)
        shutil.copyfile(mask_path, dest_mask_path)
