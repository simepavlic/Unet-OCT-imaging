from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix


def dice_loss(result_path, label_path):
    result_path = np.matrix.round(np.asarray(Image.open(result_path))/16000)
    result_path = result_path.flatten()
    label_path = np.asarray(Image.open((label_path)))/60
    label_path = label_path.flatten()
    return confusion_matrix(label_path, result_path).ravel().reshape(5,5)


