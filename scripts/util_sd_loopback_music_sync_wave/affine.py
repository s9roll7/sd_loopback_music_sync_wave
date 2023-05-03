# https://imagingsolution.net/program/python/opencv-python/opencv_python_affine_transformation/

import cv2
import numpy as np
from PIL import Image

def scaleMatrix(scale):
    mat = identityMatrix()
    mat[0,0] = scale
    mat[1,1] = scale

    return mat

def scaleXYMatrix(sx, sy):
    mat = identityMatrix()
    mat[0,0] = sx
    mat[1,1] = sy

    return mat

def translateMatrix(tx, ty):
    mat = identityMatrix()
    mat[0,2] = tx
    mat[1,2] = ty

    return mat

def rotateMatrix(deg):
    mat = identityMatrix()
    rad = np.deg2rad(deg)
    sin = np.sin(rad)
    cos = np.cos(rad)

    mat[0,0] = cos
    mat[0,1] = -sin
    mat[1,0] = sin
    mat[1,1] = cos

    return mat

def scaleAtMatrix(scale, cx, cy):
    mat = translateMatrix(-cx, -cy)
    mat = scaleMatrix(scale).dot(mat)
    mat = translateMatrix(cx, cy).dot(mat)

    return mat

def rotateAtMatrix(deg, cx, cy):
    mat = translateMatrix(-cx, -cy)
    mat = rotateMatrix(deg).dot(mat)
    mat = translateMatrix(cx, cy).dot(mat)

    return mat

def afiinePoint(mat, px, py):

    srcPoint = np.array([px, py, 1])

    return mat.dot(srcPoint)[:2]

def inverse(mat):
    return np.linalg.inv(mat)

def identityMatrix():
    return np.eye(3, dtype = np.float32)

# https://github.com/eborboihuc/rotate_3d/blob/master/image_transformer.py
def getProjectionMatrix(theta, phi, gamma, dx, dy, scale, cx, cy, width, height):

    theta, phi, gamma = np.deg2rad([theta, phi, gamma])
    
    d = np.sqrt(height**2 + width**2)
    focal = d / (2 * np.sin(gamma) if np.sin(gamma) != 0 else 1)
    dz = focal / scale

    w = width
    h = height
    f = focal

    # Projection 2D -> 3D matrix
    A1 = np.array([ [1, 0, -cx],
                    [0, 1, -cy],
                    [0, 0, 1],
                    [0, 0, 1]])
    
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([ [1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]])
    
    RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                    [0, 1, 0, 0],
                    [np.sin(phi), 0, np.cos(phi), 0],
                    [0, 0, 0, 1]])
    
    RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                    [np.sin(gamma), np.cos(gamma), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix
    T = np.array([  [1, 0, 0, dx],
                    [0, 1, 0, dy],
                    [0, 0, 1, dz],
                    [0, 0, 0, 1]])

    # Projection 3D -> 2D matrix
    A2 = np.array([ [f, 0, cx, 0],
                    [0, f, cy, 0],
                    [0, 0, 1, 0]])

    # Final transformation matrix
    return np.dot(A2, np.dot(T, np.dot(R, A1)))



def AffineImage(img:Image, x,y,angle,scale,cx,cy,angle_x,angle_y):
    if x == 0 and y == 0 and angle == 0 and scale == 1 and angle_x == 0 and angle_y == 0:
        return img

    img_array = np.asarray(img)
    h, w, c = img_array.shape

    x = w * x
    y = h * y
    cx = w * cx
    cy = h * (1-cy)
    '''
    matAffine = translateMatrix(-cx, -cy)
    matAffine = scaleMatrix(scale).dot(matAffine)
    matAffine = rotateMatrix(-angle).dot(matAffine)
    matAffine = translateMatrix(cx, cy).dot(matAffine)
    matAffine = translateMatrix(x, -y).dot(matAffine)
    '''
    matAffine = getProjectionMatrix(angle_x, angle_y, angle, x, -y, scale, cx,cy, w, h)

    if scale >= 1.0:
        interpolation = interpolation=cv2.INTER_CUBIC
    else:
        interpolation = interpolation=cv2.INTER_AREA

#    img_array = cv2.warpAffine(img_array, matAffine[:2,], (w, h), borderMode=cv2.BORDER_CONSTANT, flags=interpolation)
    img_array = cv2.warpPerspective(img_array, matAffine, (w, h), borderMode=cv2.BORDER_CONSTANT, flags=interpolation)

    return Image.fromarray(img_array)
