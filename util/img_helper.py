import cv2
import numpy as np

import random as random_gen
from time import sleep


def merge_images(left_image, right_image):
    right_image = cv2.resize(right_image, (right_image.shape[1] * left_image.shape[0] // right_image.shape[0],
                                           left_image.shape[0]))

    img = np.concatenate((left_image, right_image), axis=1)
    return img


def resize_image(image):
    scale = 240.0 / float(max(image.shape[0], image.shape[1]))
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)


def load_img(filename, shape=(384, 512, 3)):
    loading_success = False
    tries = 10
    while not loading_success:
        try:
            image = cv2.imread(str(filename))
            image = np.asarray(image, dtype=np.uint8)  # Fix for failing img loading
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            loading_success = True
        except:  # May also not have been an image file
            print('Failed to load image {}, backing off and retrying.'.format(filename))
            sleep(random_gen.uniform(0.001, 0.01))  # Back off from concurrent image access
            tries = tries - 1
            if tries == 0:
                image = np.random.randint(255, size=shape).astype(np.uint8)
                loading_success = True
                # raise ValueError('Could not load image {}.'.format(filename))
    return resize_image(image)


def load_imgs(filenames):
    imgs = []
    for filename in filenames:
        img = load_img(filename)
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs


def put_text(text, y, image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left = (10, y)
    scale = 0.5
    color = (255, 0, 0)
    line_type = 2
    return cv2.putText(image, text, bottom_left, font, scale, color, line_type)


def put_text_vi(text, image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left = (-3, -1)
    scale = 20
    color = (255, 0, 0)
    line_type = 2
    return cv2.putText(image, text, bottom_left, font, scale, color, line_type)


def save_img(filename, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(filename), image)
