import cv2
import numpy as np
from numpy import sin, cos, tan
import pandas as pd
from os import listdir
from os.path import isfile, join
from datetime import datetime


SP = 15 # size of gaussian kernal and std to apply
PERCENTILE = .99
LAPLACE_SMOOTHING_PARAM = .001


def align_image(im1, im2):
    '''
    Aligns im2 to im1 using opencv
    '''
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)


    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations,
                termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(
        im1_gray, im2_gray,
        warp_matrix, warp_mode,
        criteria
    )

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(
            im2, warp_matrix,
            (sz[1],sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(
            im2, warp_matrix,
            (sz[1],sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

    return im2_aligned


def align_images(images):
    master = images[0]
    aligned_images = [master]
    for idx, image in enumerate(images[1:]):
        print(f'aligning image: {idx + 1}, {datetime.now()}')
        image = align_image(master, image)
        aligned_images.append(image)

    return aligned_images


def resize_img(img, scale_percent=10):
    y_old = img.shape[0]
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    y_new = resized.shape[0]
    return resized, y_new / y_old


def take_laplacian(img):
    kernel_size = 3
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    img = cv2.GaussianBlur(img,(3,3),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray_lap = cv2.Laplacian(
        gray,ddepth,ksize = kernel_size,scale = scale,delta = delta
    )
    dst = cv2.convertScaleAbs(gray_lap)

    return dst


def main(path):

    files = [f for f  in listdir(path) if isfile(join(path, f))]
    images = [cv2.imread(join(path, f)) for f in files]
    images = align_images(images)

    all_weights = []

    for idx, image in enumerate(images):
        shape = image.shape[:2]
        img, scale = resize_img(image, scale_percent=10)

        lap = take_laplacian(img)

        threshold = pd.DataFrame(lap.reshape(-1,)).quantile(PERCENTILE).values[0]

        bools = (lap > threshold)

        weights = cv2.GaussianBlur(
            bools.astype(float),(SP,SP), sigmaX=SP, sigmaY=SP
        )

        # laplace smoothing
        weights = weights + LAPLACE_SMOOTHING_PARAM

        # normalize
        weights /= np.max(weights)

        # back to full scale
        weights = cv2.resize(weights, dsize=(shape[1], shape[0]))

        all_weights.append(weights)

	# combine weights in to one array, 3rd dim is the index of image
    weights = np.stack(all_weights, axis=2)

	# normalize such that sum down 3rd dim = 1
    sums = np.sum(weights, axis=2)
    weights /= sums[:, :, np.newaxis]

    output_image = np.zeros((shape[0], shape[1], 3))

    for idx, image in enumerate(images):
        weight = weights[:, :, idx]
        output_image += image * weight[:, :, np.newaxis]

    output_image = output_image.astype(int)
    lsp = str(LAPLACE_SMOOTHING_PARAM).split('.')[1]
    pct = str(PERCENTILE).split('.')[1]
    filepath = path + 'output_' + lsp + '_' + pct + '_' + str(SP) + '.jpg'
    cv2.imwrite(filepath, output_image)
    print(f'saved to: {filepath}')


if __name__ == "__main__":

    path = '/home/lewis/Pictures/pushpin2/'
    main(path)

