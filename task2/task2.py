from skimage.io import imread, imshow, imsave
import numpy as np
import sys
from matplotlib import pyplot as plt


def to_gray(img):
    """RGB to Gray

    Parameters
    ----------
    img: 3darray
        image
    Returns
    -------
    gray_img: 2darray
        gray scaled image
    """
    gray_img = img[:,:,0]*0.2989 + img[:,:,1]*0.5870 + img[:,:,2]*0.1140

    return gray_img.astype(np.uint8)


def gradient(s, img, out_path=''):
    """Gradient

    Parameters
    ----------
    s: int
        1/3 half the width of the filter
    img: ndarray
        input image
    out_path: string, optional
        path of the output image

    Returns
    -------
    new: ndarray
        gradient of image
    """

    s = float(s)
    sigma = round(s + 0.01)

    if len(img.shape) == 3:
        img = to_gray(img)

    d_x = np.zeros(img.shape, dtype=np.float64)
    d_y = np.zeros(img.shape, dtype=np.float64)

    for i in range(3*sigma):  # Extrapolation
        img = np.vstack((img[0, np.newaxis, :], img, img[-1, np.newaxis, :]))
        img = np.hstack((img[:, 0, np.newaxis], img, img[:, -1, np.newaxis]))

    # Find G'-filter matrix
    x, y = np.meshgrid(np.linspace(-3*sigma,3*sigma,3*sigma*2+1), np.linspace(3*sigma,-3*sigma,3*sigma*2+1))
    gx_m = -x*np.exp((-x ** 2 - y ** 2) / (2 * s ** 2)) / (2 * np.pi * s ** 4)
    gy_m = -y*np.exp((-x ** 2 - y ** 2) / (2 * s ** 2)) / (2 * np.pi * s ** 4)

    # Image filtering
    for i in range(d_x.shape[0]):
        for j in range(d_x.shape[1]):
            d_x[i, j] = np.sum(img[i:i + 3*sigma*2+1, j:j + 3*sigma*2+1]*gx_m)
            d_y[i, j] = np.sum(img[i:i + 3*sigma*2+1, j:j + 3*sigma*2+1]*gy_m)



    # Gradient
    grad = np.hypot(d_x, d_y)

    # Contrasting
    min = grad.min()
    max = grad.max()
    grad = (grad - min) / (max - min) * 255
    grad = grad.astype(np.uint8)

    if out_path != '':
        imsave(out_path, grad)

    return grad, d_x, d_y


def dir(s, in_path, out_path=''):
    """Gradient Direction

    Parameters
    ----------
    s: int
        1/3 half the width of the filter
    in_path: string
        path of the input image
    out_path: string, optional
        path of the output image
    gaussian: bool
        whether to remove noise?
    Returns
    -------
    new: ndarray
        filtered image
    """
    img = imread(in_path)

    grad, d_x, d_y = gradient(s, img)
    new = np.zeros(d_x.shape, dtype=np.uint8)
    tan = np.arctan2(d_y, d_x) * 180/np.pi
    tan[tan<0] += 180
    for i in range(d_x.shape[0]):
        for j in range(d_x.shape[1]):
            if d_x[i,j] == d_y[i,j] == 0:
                new[i,j] = 0
            elif 0 <= tan[i,j] < 22.5 or 157.5 <= tan[i,j] <= 180:
                new[i,j] = 64
            elif 22.5 <= tan[i,j] < 67.5:
                new[i,j] = 192
            elif 67.5 <= tan[i, j] < 112.5:
                new[i,j] = 128
            elif 112.5 <= tan[i, j] < 157.5:
                new[i,j] = 255
    if out_path != '':
        imsave(out_path, new)
    return grad, new


def nonmax(s, in_path, out_path='', contrasting=True):
    """Non maximum suppression

    Parameters
    ----------
    s: int
        1/3 half the width of the filter
    in_path: string
        path of the input image
    out_path: string, optional
        path of the output image
    contrasting: bool
        whether to do contrasting?
    Returns
    -------
    new: ndarray
        filtered image
    """
    grad, D = dir(s, in_path)
    new = np.zeros(grad.shape, dtype=np.uint8)
    grad = np.vstack((grad[0, np.newaxis, :], grad, grad[-1, np.newaxis, :]))
    grad = np.hstack((grad[:, 0, np.newaxis], grad, grad[:, -1, np.newaxis]))
    for i in range(1, new.shape[0]+1):
        for j in range(1, new.shape[1]+1):
            a = b = 255
            if D[i-1,j-1] == 64:
                a,b = grad[i,j-1], grad[i,j+1]
            elif D[i-1,j-1] == 128:
                a,b = grad[i-1,j], grad[i+1,j]
            elif D[i-1,j-1] == 192:
                a,b = grad[i+1,j-1], grad[i-1,j+1]
            elif D[i-1,j-1] == 255:
                a,b = grad[i-1,j-1], grad[i+1,j+1]

            if grad[i,j] >= a and grad[i,j] >= b:
                new[i-1,j-1] = grad[i,j]

    if bool(contrasting):
        min = new.min()
        max = new.max()
        new = (new-min)/(max-min)*255
    new = new.astype(np.uint8)
    if out_path != '':
        imsave(out_path, new)
    return new


def canny(s=2, thr_high=0.2, thr_low=0.02, in_path='', out_path=''):
    """Canny edge detection

    Parameters
    ----------
    s: int
        1/3 half the width of the filter
    thr_high: float
        high threshold ratio
    thr_low: float
        low threshold ratio
    in_path: string
        path of the input image
    out_path: string, optional
        path of the output image
    Returns
    -------
    new: ndarray
        filtered image
    """
    img = nonmax(s, in_path, contrasting=False)

    high = img.max() * float(thr_high)
    low = high * float(thr_low)

    img = np.where(img > high, 255, img)
    img = np.where(img < low, 0, img)
    img = np.vstack((img[0, np.newaxis, :], img, img[-1, np.newaxis, :]))
    img = np.hstack((img[:, 0, np.newaxis], img, img[:, -1, np.newaxis]))

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if low <= img[i,j] <= high:
                if (img[i-1:i+2,j-1:j+2] == 255).any():
                    img[i,j] = 255
                else:
                    img[i,j] = 0
    img = np.delete(img,[0,img.shape[1]-1],axis=1)
    img = np.delete(img,[0,img.shape[0]-1],axis=0)

    if out_path != '':
        imsave(out_path, img)
    return img


def bilateral(sigma_d=5, sigma_r=20, in_path='', out_path=''):
    """Bilateral filter

    Parameters
    ----------
    sigma_d: float
        1/3 half the width of the filter
    sigma_r: float
    in_path: string
        path of the input image
    out_path: string, optional
        path of the output image

    Returns
    -------
    new: ndarray
        filtered image
    """
    # t = time.time() #time_start
    s_d = float(sigma_d)
    s_r = float(sigma_r)
    sigma = round(s_d+0.01)
    img = imread(in_path)

    if len(img.shape) == 3:
        img = to_gray(img)

    new = np.zeros(img.shape, dtype=np.float16)

    for i in range(3*sigma): # Extrapolation
        img = np.vstack((img[0, np.newaxis, :], img, img[-1, np.newaxis, :]))
        img = np.hstack((img[:, 0, np.newaxis], img, img[:, -1, np.newaxis]))

    # Find G-filter matrix
    x, y = np.meshgrid(np.linspace(-3*sigma,3*sigma,3*sigma*2+1), np.linspace(3*sigma,-3*sigma,3*sigma*2+1))
    g_m = np.exp((-x ** 2 - y ** 2) / (2 * s_d**2))

    # Image filtering
    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            batch = img[i:i+3*sigma*2+1, j:j+3*sigma*2+1].astype(int)
            intensity = batch[batch.shape[0]//2, batch.shape[0]//2].astype(int)
            r_m = np.exp(-(batch-intensity) ** 2 / (2 * s_r**2))
            new[i, j] = np.sum(batch * g_m * r_m) / np.sum(g_m * r_m)
    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)

    if out_path != '':
        imsave(out_path, new)

    return new


if __name__ == '__main__':
    if len(sys.argv) > 2:
        globals()[sys.argv[1]](*[sys.argv[i] for i in range(2, len(sys.argv))])