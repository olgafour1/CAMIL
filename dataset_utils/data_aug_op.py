import random
import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import tensorflow_addons as tfa

class RandomColorAffine(tf.keras.layers.Layer):
    def __init__(self, zoom_factor, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.zoom_factor = zoom_factor
        self.brightness = brightness
        self.jitter = jitter

    def call(self, images, training=True):
        images = self._random_apply(tf.image.flip_left_right, images, p=0.5)
        images = self._random_apply(tfa.image.translate, images, translations=[self.zoom_factor / 2,
                                                                               self.zoom_factor / 2], p=0.2)
        if training:

            batch_size = tf.shape(images)[0]

            # Same for all colors
            brightness_scales = 1 + tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness
            )
            # Different for all colors
            jitter_matrices = tf.random.uniform(
                (batch_size, 1, 3, 3), minval=-self.jitter, maxval=self.jitter
            )

            color_transforms = (
                    tf.eye(3, batch_shape=[batch_size, 1]) * brightness_scales
                    + jitter_matrices
            )
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images

    def _random_apply(self, func, x, p, **kwargs):
        return tf.cond(
            tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                    tf.cast(p, tf.float32)),
            lambda: func(x, **kwargs),
            lambda: x)

def random_translation(img):
    t_x = np.random.randint(img.shape[0] / 2, size=1)

    t_y = np.random.randint(img.shape[0] / 2, size=1)
    M = np.float32([[1, 0, t_x], [0, 1, t_y]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted


def random_flip_img(img, horizontal_chance=0, vertical_chance=0):
    """

    Parameters
    ----------
    img:  np.ndarray containing an image
    horizontal_chance: the probability of flipping horizontally the image
    vertical_chance: the probability of flipping vertically the image

    Returns
    -------
    img: flipped image
    """
    flip_horizontal = False
    if random.random() < horizontal_chance:
        flip_horizontal = True

    flip_vertical = False
    if random.random() < vertical_chance:
        flip_vertical = True

    if not flip_horizontal and not flip_vertical:
        return img

    flip_val = 1
    if flip_vertical:
        flip_val = -1 if flip_horizontal else 0

    if not isinstance(img, list):
        res = cv2.flip(img, flip_val)
    else:
        res = []
        for img_item in img:
            img_flip = cv2.flip(img_item, flip_val)
            res.append(img_flip)
    return res


def random_rotate_img(images):
    """

    Parameters
    ----------
    images: np.ndarray of an image

    Returns
    -------
    img_inst: a randomly rotated image
    """
    rand_roat = np.random.randint(4, size=1)
    angle = 90 * rand_roat
    center = (images.shape[0] / 2, images.shape[1] / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle[0], scale=1.0)

    img_inst = cv2.warpAffine(images, rot_matrix, dsize=images.shape[:2], borderMode=cv2.BORDER_CONSTANT)

    return img_inst


def hematoxylin_eosin_aug(image, seed=None):
    """
    "Quantification of histochemical staining by color deconvolution"
    Arnout C. Ruifrok, Ph.D. and Dennis A. Johnston, Ph.D.
    http://www.math-info.univ-paris5.fr/~lomn/Data/2017/Color/Quantification_of_histochemical_staining.pdf
    Performs random hematoxylin-eosin augmentation
    """
    D = np.array([[1.88, -0.07, -0.60],
                  [-1.02, 1.13, -0.48],
                  [-0.55, -0.13, 1.57]])
    M = np.array([[0.65, 0.70, 0.29],
                  [0.07, 0.99, 0.11],
                  [0.27, 0.57, 0.78]])
    Io = 240

    h, w, c = image.shape
    OD = -np.log10((image.astype("uint16") + 1) / Io)
    C = np.dot(D, OD.reshape(h * w, c).T).T
    r = np.ones(3)
    # r[:2] = np.random.RandomState(seed).uniform(low=low, high=high, size=2)
    r[:2] = np.random.RandomState(seed).normal(loc=1.0, scale=0.02, size=2)

    img_aug = np.dot(C, M) * r

    img_aug = Io * np.exp(-img_aug * np.log(10)) - 1
    img_aug = img_aug.reshape(h, w, c).clip(0, 255).astype("uint8")
    return img_aug


def normalize(image, target=None):
    """Normalizing function we got from the cedars-sinai medical center"""
    if target is None:
        target = np.array([[57.4, 15.84], [39.9, 9.14], [-22.34, 6.58]])

    whitemask = color.rgb2gray(image)
    whitemask = whitemask > (215 / 255)

    imagelab = color.rgb2lab(image)

    imageL, imageA, imageB = [imagelab[:, :, i] for i in range(3)]

    # mask is valid when true
    imageLM = np.ma.MaskedArray(imageL, whitemask)
    imageAM = np.ma.MaskedArray(imageA, whitemask)
    imageBM = np.ma.MaskedArray(imageB, whitemask)

    ## Sometimes STD is near 0, or 0; add epsilon to avoid div by 0 -NI
    epsilon = 1e-11

    imageLMean = imageLM.mean()
    imageLSTD = imageLM.std() + epsilon

    imageAMean = imageAM.mean()
    imageASTD = imageAM.std() + epsilon

    imageBMean = imageBM.mean()
    imageBSTD = imageBM.std() + epsilon

    # normalization in lab
    imageL = (imageL - imageLMean) / imageLSTD * target[0][1] + target[0][0]
    imageA = (imageA - imageAMean) / imageASTD * target[1][1] + target[1][0]
    imageB = (imageB - imageBMean) / imageBSTD * target[2][1] + target[2][0]

    imagelab = np.zeros(image.shape)
    imagelab[:, :, 0] = imageL
    imagelab[:, :, 1] = imageA
    imagelab[:, :, 2] = imageB

    # Back to RGB space
    returnimage = color.lab2rgb(imagelab)
    returnimage = np.clip(returnimage, 0, 1)
    returnimage *= 255

    # Replace white pixels
    returnimage[whitemask] = image[whitemask]
    return returnimage.astype(np.uint8)


def add_gaussian_noise(img):
    img = cv2.blur(img, (1, 1))
    return img


def normalize_staining(image):
    Io = 240
    beta = 0.15
    alpha = 1
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = image.shape
    img = image.reshape(h * w, c)
    OD = -np.log((img.astype("uint16") + 1) / Io)
    ODhat = OD[(OD >= beta).all(axis=1)]
    W, V = np.linalg.eig(np.cov(ODhat, rowvar=False))

    Vec = -V.T[:2][::-1].T
    That = np.dot(ODhat, Vec)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if vMin[0] > vMax[0]:
        HE = np.array([vMin, vMax])
    else:
        HE = np.array([vMax, vMin])

    HE = HE.T
    Y = OD.reshape(h * w, c).T

    C = np.linalg.lstsq(HE, Y, rcond=None)
    maxC = np.percentile(C[0], 99, axis=1)

    C = C[0] / maxC[:, None]
    C = C * maxCRef[:, None]
    Inorm = Io * np.exp(-np.dot(HERef, C))
    Inorm = Inorm.T.reshape(h, w, c).clip(0, 255).astype("uint8")

    return Inorm


def elastic_transform(image, alpha_range=(10, 20), sigma=4, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

   # Arguments
       image: Numpy array with shape (height, width, channels).
       alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
           Controls intensity of deformation.
       sigma: Float, sigma of gaussian filter that smooths the displacement fields.
       random_state: `numpy.random.RandomState` object for generating displacement fields.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def color_jitter(image, img_brightness=0.2, img_contrast=0.2, img_saturation=0.2):
    """
    Perfrom a color jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """

    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))

        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                f = random.uniform(-img_brightness, img_brightness)
                image = np.clip(image + f, 0., 1.).astype(np.float32)

            elif jitter[order[idx]] == "contrast":
                f = random.uniform(1 - img_contrast, 1 + img_contrast)
                image = np.clip(image * f, 0., 1.)

            elif jitter[order[idx]] == "saturation":
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                f = random.uniform(-img_saturation, img_saturation)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] + f, 0., 1.)
                image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image


def random_crop(image, crop_chance=0.5):
    if random.random() < crop_chance:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        begin, size, bbox_for_draw = sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(0.75, 1.33),
            area_range=(0.5, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)

        top_y, left_x, bottom_y, right_x = (tf.squeeze(bbox_for_draw).numpy()) * image.shape[0]

        cropped = image[int(top_y):int(bottom_y), int(left_x):int(right_x)]

        top = 0
        bottom = image.shape[0] - cropped.shape[0]
        left = 0
        right = image.shape[1] - cropped.shape[1]

        value = [0, 0, 0]

        image = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value)
    return image


def preprocess_input(x):

        x /= 127.5
        x -= 1.
        x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]

        return x
