import cv2
import numpy as np


def expand_eyebrows(lmrks, eyebrows_expand_mod=1.0):

    lmrks = np.array(lmrks.copy(), dtype=np.int32)

    # Top of the eye arrays
    bot_l = lmrks[[37, 38, 39, 40]]
    bot_r = lmrks[[43, 44, 45, 46]]

    # Extrapolate to five points
    bot_l = np.array(
        [
            2 * bot_l[0] - bot_l[1],
            bot_l[0],
            bot_l[1],
            bot_l[2],
            2 * bot_l[3] - bot_l[2],
        ]
    )
    bot_r = np.array(
        [
            2 * bot_r[0] - bot_r[1],
            bot_r[0],
            bot_r[1],
            bot_r[2],
            2 * bot_r[3] - bot_r[2],
        ]
    )

    # Eyebrow arrays
    top_l = lmrks[[18, 19, 20, 21, 22]]
    top_r = lmrks[[23, 24, 25, 26, 27]]

    # Adjust eyebrow arrays
    lmrks[[18, 19, 20, 21, 22]] = top_l + eyebrows_expand_mod * 0.5 * (top_l - bot_l)
    lmrks[[23, 24, 25, 26, 27]] = top_r + eyebrows_expand_mod * 0.5 * (top_r - bot_r)
    return lmrks


def get_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Get face mask of image size using given landmarks of person
    """

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)

    points = np.array(landmarks, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 255)

    return mask


def face_mask_static(
    image: np.ndarray, landmarks: np.ndarray, landmarks_tgt: np.ndarray, params=None
) -> np.ndarray:
    """
    Get the final mask, using landmarks and applying blur
    """
    if params is None:

        left = np.sum(
            (
                landmarks[1][0] - landmarks_tgt[1][0],
                landmarks[4][0] - landmarks_tgt[4][0],
                landmarks[3][0] - landmarks_tgt[3][0],
            )
        )
        right = np.sum(
            (
                landmarks_tgt[17][0] - landmarks[17][0],
                landmarks_tgt[14][0] - landmarks[14][0],
                landmarks_tgt[16][0] - landmarks[16][0],
            )
        )

        offset = max(left, right)

        if offset > 6:
            erode = 15
            sigmaX = 15
            sigmaY = 10
        elif offset > 3:
            erode = 10
            sigmaX = 10
            sigmaY = 8
        elif offset < -3:
            erode = -5
            sigmaX = 5
            sigmaY = 10
        else:
            erode = 5
            sigmaX = 5
            sigmaY = 5

    else:
        erode = params[0]
        sigmaX = params[1]
        sigmaY = params[2]

    if erode == 15:
        eyebrows_expand_mod = 2.7
    elif erode == -5:
        eyebrows_expand_mod = 0.5
    else:
        eyebrows_expand_mod = 2.0
    landmarks = expand_eyebrows(landmarks, eyebrows_expand_mod=eyebrows_expand_mod)

    mask = get_mask(image, landmarks)
    mask = erode_and_blur(mask, erode, sigmaX, sigmaY, True)

    if params is None:
        return mask / 255, [erode, sigmaX, sigmaY]

    return mask / 255


def erode_and_blur(mask_input, erode, sigmaX, sigmaY, fade_to_border=True):
    mask = np.copy(mask_input)

    if erode > 0:
        kernel = np.ones((erode, erode), "uint8")
        mask = cv2.erode(mask, kernel, iterations=1)

    else:
        kernel = np.ones((-erode, -erode), "uint8")
        mask = cv2.dilate(mask, kernel, iterations=1)

    if fade_to_border:
        clip_size = sigmaY * 2
        mask[:clip_size, :] = 0
        mask[-clip_size:, :] = 0
        mask[:, :clip_size] = 0
        mask[:, -clip_size:] = 0

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigmaX, sigmaY=sigmaY)

    return mask
