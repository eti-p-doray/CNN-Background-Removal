import os
import cv2
import random
import numpy
import math


def image_fill(img, size, value):

    border = [math.ceil((size[0] - img.shape[0])/2),
              math.floor((size[0] - img.shape[0])/2),
              math.ceil((size[1] - img.shape[1])/2),
              math.floor((size[1] - img.shape[1])/2)]
    return cv2.copyMakeBorder(img,border[0],border[1],border[2],border[3],cv2.BORDER_CONSTANT,value=value)


def combine_object_background(object_file, background_file, output_name):
    border = 20
    size = [960, 720]

    foreground = cv2.imread(object_file, cv2.IMREAD_UNCHANGED)
    if foreground is None:
        return False

    ratio = numpy.amin(numpy.divide(
            numpy.subtract(size, [2*border, 2*border]), foreground.shape[0:2]))
    forground_size = numpy.floor(numpy.multiply(foreground.shape[0:2], ratio)).astype(int)
    foreground = cv2.resize(foreground, (forground_size[1], forground_size[0]))
    foreground = image_fill(foreground,size,[0,0,0,0])

    foreground = foreground.astype(float)
    cv2.normalize(foreground, foreground, 0.0, 1.0, cv2.NORM_MINMAX)
    alpha = cv2.split(foreground)[3]

    #foreground = cv2.imread(object_file, cv2.IMREAD_COLOR)
    background = cv2.imread(background_file)
    if background is None:
        return False

    ratio = numpy.amax(numpy.divide(foreground.shape[0:2], background.shape[0:2]))
    background_size = numpy.ceil(numpy.multiply(background.shape[0:2], ratio)).astype(int)
    #print(numpy.multiply(background.shape[0:2], ratio).astype(int))
    background = cv2.resize(background, (background_size[1], background_size[0]))
    background = background[0:foreground.shape[0], 0:foreground.shape[1]]
    background = background.astype(float)

    for i in range(0, 3):
        foreground[:,:,i] = numpy.multiply(alpha, foreground[:,:,i]*255)
        background[:,:,i] = numpy.multiply(1.0 - alpha, background[:,:,i])
    outImage = numpy.add(foreground[:,:,0:3], background)

    cv2.imwrite(output_name, outImage)

    return True


def generate_trimap(object_file, trimap_name):
    border = 20
    size = [960, 720]

    foreground = cv2.imread(object_file, cv2.IMREAD_UNCHANGED)
    if foreground is None:
        return False
    alpha = cv2.split(foreground)[3]

    ratio = numpy.amin(numpy.divide(
            numpy.subtract(size, [2*border, 2*border]), alpha.shape[0:2]))
    forground_size = numpy.floor(numpy.multiply(alpha.shape[0:2], ratio)).astype(int)
    alpha = cv2.resize(alpha, (forground_size[1], forground_size[0]))
    alpha = image_fill(alpha,size,[0,0,0,0])

    alpha = alpha.astype(float)
    cv2.normalize(alpha, alpha, 0.0, 1.0, cv2.NORM_MINMAX)

    _, inner_map = cv2.threshold(alpha, 0.999, 255, cv2.THRESH_BINARY)
    _, outer_map = cv2.threshold(alpha, 0.001, 255, cv2.THRESH_BINARY)

    inner_map = cv2.erode(inner_map, numpy.ones((5,5),numpy.uint8), iterations = 3)
    outer_map = cv2.dilate(outer_map, numpy.ones((5,5),numpy.uint8), iterations = 3)

    cv2.imwrite(trimap_name, inner_map + (outer_map - inner_map) /2)

    foreground = cv2.imread(object_file, cv2.IMREAD_UNCHANGED)

def generate_target(object_file, target_name):
    border = 20
    size = [960, 720]

    foreground = cv2.imread(object_file, cv2.IMREAD_UNCHANGED)
    if foreground is None:
        return False
    cv2.normalize(foreground, foreground, 0, 255, cv2.NORM_MINMAX)
    foreground = foreground.astype(numpy.uint8)

    ratio = numpy.amin(numpy.divide(
            numpy.subtract(size, [2*border, 2*border]), foreground.shape[0:2]))
    forground_size = numpy.floor(numpy.multiply(foreground.shape[0:2], ratio)).astype(int)
    foreground = cv2.resize(foreground, (forground_size[1], forground_size[0]))
    foreground = image_fill(foreground,size,[0,0,0,0])

    cv2.imwrite(target_name, foreground)


def build_dataset(object_dir, background_dir, input_dir, trimap_dir, target_dir):
    object_filenames = os.listdir(object_dir)
    background_filenames = os.listdir(background_dir)

    for i, object_file in enumerate(object_filenames):
        generate_trimap(
            os.path.join(object_dir, object_file),
            os.path.join(trimap_dir, str(i) + '_trimap.jpg'))
        generate_target(
            os.path.join(object_dir, object_file),
            os.path.join(target_dir, str(i) + '.png'))

        backgrounds = random.sample(background_filenames, 20)
        for j, background_file in enumerate(backgrounds):
            print(i, j, object_file, background_file)
            combine_object_background(os.path.join(object_dir, object_file),
              os.path.join(background_dir, background_file),
              os.path.join(input_dir, str(i) + '_' + str(j) + '.jpg'))


if __name__ == "__main__":
    object_dir = os.path.join("data", "matting", "portrait transparent background")
    background_dir = os.path.join("data", "matting", "texture background")
    input_dir = os.path.join("data", "matting", "input")
    trimap_dir = os.path.join("data", "matting", "trimap")
    target_dir = os.path.join("data", "matting", "target")

    if not os.path.isdir(input_dir):
        os.makedirs(input_dir)
    if not os.path.isdir(trimap_dir):
        os.makedirs(trimap_dir)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    build_dataset(object_dir, background_dir, input_dir, trimap_dir, target_dir)
