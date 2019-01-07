'''
Preprocess the observation from the agent.
'''

import cv2
import numpy as np

CROP_OFFSET = 0
def preprocess_image(image, resize_width=84, resize_height=84, resize_mode='scale'):
    '''do preprocess to images'''
    #image,h,w,c
    resized_image = resize_image(image, resize_width, resize_height, resize_mode)
    return gray_image(resized_image)

def resize_image(image, resize_width, resize_height, resize_mode):
    '''Appropriately resize a single image '''
    height = image.shape[0]
    width = image.shape[1]
    #print(height,width)
    if resize_mode == 'crop':
        #resize keeping aspect ratio
        resize_height = int(round(
            float(height) * resize_width / width))

        resized = cv2.resize(image,(resize_width,resize_height),interpolation=cv2.INTER_LINEAR)

        #Crop the part we want
        if not CROP_OFFSET:
            crop_y_cutoff = resize_height - CROP_OFFSET - resize_height
            cropped = resized[crop_y_cutoff:crop_y_cutoff + resize_height, :]
            return cropped

        return resized
    elif resize_mode == 'scale':
        return cv2.resize(image,(resize_width,resize_height),interpolation=cv2.INTER_LINEAR)

    else:
        raise ValueError('Unrecognized image resize method.')


def gray_image(image):
    # h, w, c instead of c, h, w
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    w = gray.shape[1]
    return gray.reshape(h,w,-1)
