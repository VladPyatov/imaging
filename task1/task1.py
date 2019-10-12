from skimage.io import imread, imshow, imsave
from matplotlib import pyplot as plt
import numpy as np
import sys
# import time


def mirror(axis, in_path, out_path=''):
    """Image mirroring

    Parameters
    ----------
    axis: string
        x or y
    in_path: string
        path of the input image
    out_path: string, optional
        path of the output image

    Returns
    -------
    img: ndarray
        mirrored image
    """

    img = imread(in_path)
    img_c = img.copy()

    if axis == 'y':
        for i in range(img.shape[0]):
            for j in range(img.shape[1] // 2):
                img[i, j], img[i, -(j+1)] = img_c[i, -(j+1)], img_c[i, j]
    elif axis == 'x':
        for j in range(img.shape[1]):
            for i in range(img.shape[0] // 2):
                img[i, j], img[-(i+1), j] = img_c[-(i+1), j], img_c[i, j]

    if out_path != '':
        imsave(out_path, img)
    return img


def rotate(s, angle, in_path, out_path=''):
    """Image rotation

    Parameters
    ----------
    s: string
        cw or ccw - how to rotate
    angle: int
        angle of rotation
    in_path: string
        path of the input image
    out_path: string, optional
        path of the output image

    Returns
    -------
    n_img: ndarray
        rotated image
    """
    img = imread(in_path)
    angle = int(angle)
    n_img = np.zeros((img.shape[1 if angle//90%2 != 0 else 0], img.shape[0 if angle//90%2 != 0 else 1], img.shape[2]),
                     dtype=np.uint8)

    if s == 'ccw' and angle//90%4 == 1 or s == 'cw' and angle//90%4 == 3:
        for i in range(img.shape[0]):
            n_img[:, i] = img[i, -1::-1]
    elif s == 'cw' and angle//90%4 == 1 or s == 'ccw' and angle//90%4 == 3:
        for i in range(img.shape[0]):
            n_img[:, i] = img[-(i+1), :]
    elif angle//90%4 == 2:
        for i in range(img.shape[0]):
            n_img[i, :] = img[-(i+1), -1::-1]
    else:
        n_img = img.copy()

    if out_path != '':
        imsave(out_path, n_img)

    return n_img


def sobel(axis, in_path, out_path=''):
    """Sobel operator

    Parameters
    ----------
    axis: string
        x or y
    in_path: string
        path of the input image
    out_path: string, optional
        path of the output image

    Returns
    -------
    new: ndarray
        filtered image
    """
    # t=time.time()
    img = imread(in_path)
    new = np.zeros(img.shape, dtype=np.int16)

    # Extrapolation
    img = np.vstack((img[0,np.newaxis,:], img, img[-1,np.newaxis,:]))
    img = np.hstack((img[:,0,np.newaxis], img, img[:,-1,np.newaxis]))

    M = np.array([[-1,-2,-1],
                  [0,0,0],
                  [1,2,1]])

    if axis == 'x':
        M = M.T

    # Image filtering
    for i in range(1, new.shape[0]+1):
        for j in range(1, new.shape[1]+1):
            nnn = img[i-1:i+2, j-1:j+2]
            new[i-1, j-1] = np.tensordot(nnn, M, axes=([0,1], [0,1]))

    new = np.clip(new + 128, 0, 255)

    new = new.astype(np.uint8)

    if out_path != '':
        imsave(out_path, new)
    # print(time.time()-t)
    return new


def median(rad, in_path, out_path=''):
    """Median operator

    Parameters
    ----------
    rad: int
        radius
    in_path: string
        path of the input image
    out_path: string, optional
        path of the output image

    Returns
    -------
    new: ndarray
        filtered image
    """
    def med(X):
        srt = sorted(np.ravel(X))
        l = len(srt)
        if not l % 2:
            return (srt[l//2-1] + srt[l//2]) / 2.0
        return srt[l//2]

    # t=time.time()
    rad = int(rad)
    ok = rad*2+1
    img = imread(in_path)

    new = np.zeros(img.shape, dtype=np.uint8)

    for i in range(rad):  # Extrapolation
        img = np.vstack((img[0, np.newaxis, :], img, img[-1, np.newaxis, :]))
        img = np.hstack((img[:, 0, np.newaxis], img, img[:, -1, np.newaxis]))

    # Image filtering
    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            if len(new.shape) == 3:
                new[i, j,0], new[i, j,1], new[i, j,2] = med(img[i:i+ok, j:j + ok,0]), med(img[i:i+ok, j:j + ok,1]), \
                                                        med(img[i:i+ok, j:j + ok,2])
            else:
                new[i, j]= med(img[i:i + ok, j:j + ok])
    if out_path != '':
        imsave(out_path, new)
    # print(time.time()-t)
    return new


def gauss(s, in_path, out_path=''):
    """Gaussian filter

    Parameters
    ----------
    s: int
        1/3 half the width of the filter
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
    s = float(s)
    sigma = round(s+0.01)
    img = imread(in_path)
    new = np.zeros(img.shape, dtype=np.float16)

    for i in range(3*sigma): # Extrapolation
        img = np.vstack((img[0, np.newaxis, :], img, img[-1, np.newaxis, :]))
        img = np.hstack((img[:, 0, np.newaxis], img, img[:, -1, np.newaxis]))

    # Find G-filter matrix
    x, y = np.meshgrid(np.linspace(-3*sigma,3*sigma,3*sigma*2+1), np.linspace(3*sigma,-3*sigma,3*sigma*2+1))
    g_m = np.exp((-x ** 2 - y ** 2) / (2 * s**2)) / (2 * np.pi * s**2)

    # Image filtering
    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            new[i, j] = np.tensordot(img[i:i+3*sigma*2+1, j:j+3*sigma*2+1], g_m, axes=([0,1], [0,1]))

    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)

    if out_path != '':
        imsave(out_path, new)
    # print(time.time() - t) #time_end
    return new


def gradient(s, in_path, out_path=''):
    """Gradient

    Parameters
    ----------
    s: int
        1/3 half the width of the filter
    in_path: string
        path of the input image
    out_path: string, optional
        path of the output image

    Returns
    -------
    new: ndarray
        filtered? image
    """
    # t = time.time() #time_start

    s = float(s)
    sigma = round(s + 0.01)
    img = imread(in_path)
    d_x = np.zeros(img.shape, dtype=np.float16)
    d_y = np.zeros(img.shape, dtype=np.float16)

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
            d_x[i, j] = np.tensordot(img[i:i+3*sigma*2+1, j:j+3*sigma*2+1], gx_m, axes=([0,1], [0,1]))

    for i in range(d_y.shape[0]):
        for j in range(d_y.shape[1]):
            d_y[i, j] = np.tensordot(img[i:i+3*sigma*2+1, j:j+3*sigma*2+1], gy_m, axes=([0,1], [0,1]))

    # Gradient
    grad = np.hypot(d_x, d_y)

    # Contrasting
    if len(grad.shape) == 3:
        R = grad[:,:,0]
        G = grad[:,:,1]
        B = grad[:,:,2]
        Y = 0.2126*R + 0.7152*G + 0.0722*B
        U = -0.0999*R - 0.3360*G + 0.4360*B
        V = 0.6150*R - 0.5586*G - 0.0563*B
    else:
        Y = grad

    new_y = np.ravel(Y.copy())
    new_y.sort()
    x_min = new_y[0]
    x_max = new_y[-1]

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (Y[i,j]-x_min)*255/(x_max-x_min)

    if len(grad.shape) == 3:
        R = Y + 1.2803*V
        G = Y - 0.2148*U - 0.3805*V
        B = Y + 2.1279*U
        grad = np.dstack((R,G,B))
    else:
        grad = Y

    grad = np.clip(grad, 0, 255)
    grad = grad.astype(np.uint8)

    if out_path != '':
        imsave(out_path, grad)

    # print(time.time() - t)  # time_end
    return grad


if __name__ == '__main__':
    if len(sys.argv) > 2:
        globals()[sys.argv[1]](*[sys.argv[i] for i in range(2, len(sys.argv))])
