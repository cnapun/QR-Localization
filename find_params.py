# pylint: disable=maybe-no-member,no-member
import numpy as np
import cv2
import math
import argparse
import itertools
import glob


def run_all(image_files_glob='Camera Localization/IMG*', pattern_file='Camera Localization/pattern.png'):
    src_pts = []
    img_pts = []
    pattern = cv2.imread(pattern_file, 0)
    for file in glob.glob(image_files_glob):
        img = cv2.imread(file, 0)
        imshape = img.shape
        pattern_contours = contour_image(pattern)
        pattern_centers = find_concentric(pattern_contours).astype(np.float32)
        where_zero = np.where(pattern == 0)[0]
        pattern_size = where_zero[-1] - where_zero[0] + 1
        print_size = 88
        thresh = thresh_image(img)
        contours = contour_image(thresh)
        img_centers = find_concentric(contours, fuzzy=False)
        if len(img_centers) != 4:
            img_centers = find_concentric(contours, fuzzy=True)
            if len(img_centers) != 4:
                raise Exception('Unable to find centers in picture')
        centers, _ = match_center_order(
            pattern_centers, img_centers, pattern, img)
        src_pts.append(np.hstack((pattern_centers / pattern_size *
                                  print_size, np.zeros((4, 1)))).astype(np.float32))
        img_pts.append(centers.astype(np.float32))
    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(src_pts, img_pts, imshape[::-1], None, None)
    out = {}
    for rvec, tvec, file in zip(rvecs, tvecs, glob.glob(image_files_glob)):
        rot = cv2.Rodrigues(rvec.astype(np.float32))[0]
        euler_vec = cv2.RQDecomp3x3(rot.T.astype(np.float32))[0]
        roll, pitch, yaw = euler_vec
        tx, ty, tz = tvec.reshape(-1)
        print(file.split('/')[-1])
        print(f'Camera Angles:\nRoll: {roll:.1f}°, Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°\nCamera Position:\n(x,y,z) = ({tx:.0f}, {ty:.0f}, {tz:.0f}) mm')
        out[file[-6:-4]] = {'roll': float(roll),
                'pitch': float(pitch),
                'yaw': float(yaw),
                'tx': float(tx),
                'ty': float(ty),
                'tz': float(tz)}
        print()
    return out


def thresh_image(img, threshold=170):
    ret, thresh = cv2.threshold(img, threshold, 255, 0)
    return cv2.floodFill(thresh, np.zeros((thresh.shape[0] + 2, thresh.shape[1] + 2)).astype(np.uint8), (0, 0), 255)[1]


def contour_image(img):
    return cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]


def fuzzy_match(cx, cy, center_dict, min_concentric=2):
    if (cx + 1, cy) in center_dict:
        center_dict[(cx + 1, cy)] = center_dict.get((cx + 1, cy), 0) + 1
    elif (cx - 1, cy) in center_dict:
        center_dict[(cx - 1, cy)] = center_dict.get((cx - 1, cy), 0) + 1
    elif (cx, cy + 1) in center_dict:
        center_dict[(cx, cy + 1)] = center_dict.get((cx, cy + 1), 0) + 1
    elif (cx, cy - 1) in center_dict:
        center_dict[(cx, cy - 1)] = center_dict.get((cx, cy - 1), 0) + 1
    else:
        center_dict[(cx, cy)] = center_dict.get((cx, cy), 0) + 1


def find_concentric(contours, min_concentric=2, fuzzy=False):
    centers = {}
    valids = []
    for c in contours:
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if not fuzzy:
            centers[(cx, cy)] = centers.get((cx, cy), 0) + 1
        else:
            fuzzy_match(cx, cy, centers, min_concentric)
    for k, v in centers.items():
        if v >= 2:
            valids.append(k)
    return np.array(valids)


def match_center_order(centers1, to_rearrange, pattern, scene):
    affs = []
    errs = []
    perms = []
    for i in itertools.permutations(to_rearrange):
        perms.append(np.array(i))
        affs.append(cv2.estimateAffine2D(perms[-1], centers1)[0])
        M = affs[-1]
        back_proj = cv2.warpAffine(scene, M, pattern.shape[::-1])
        _, back_proj = cv2.threshold(back_proj, 150, 255, 0)
        err = ((back_proj != pattern)).sum()
        errs.append(err)
    return perms[np.argmin(errs)], affs[np.argmin(errs)]


def calc_params(pattern_centers, centers2, mtx, use_cal_cam=False, img_shape=None):
    pcs = np.hstack((pattern_centers, np.zeros((4, 1))))
    _, rvec, tvec = cv2.solvePnP(pcs.astype(np.float32), centers2.astype(
        np.float32), mtx.astype(np.float32), None)
    if not use_cal_cam:
        rot = cv2.Rodrigues(rvec.astype(np.float32))[0]
        euler_vec = cv2.RQDecomp3x3(rot.T.astype(np.float32))[0]
    else:
        rvec = cv2.calibrateCamera([pcs.astype(np.float32)], [
                                   centers2.astype(np.float32)], img_shape, None, None)[3][0]
        rot = cv2.Rodrigues(rvec.astype(np.float32))[0]
        euler_vec = cv2.RQDecomp3x3(rot.T.astype(np.float32))[0]
    return euler_vec, tvec


def main(input_file, pattern_file='Camera Localization/pattern.png'):
    """

    """
    pattern = cv2.imread(pattern_file, 0)
    img = cv2.imread(input_file, 0)
    imshape = img.shape

    # Parameters from here http://www.telesens.co/2015/10/19/camera-calibration-using-opencv/
    fx = 2828
    cx = imshape[1] / 2
    fy = 2842
    cy = imshape[0] / 2
    camera_matrix = np.eye(3)
    camera_matrix[0, 0] = fx
    camera_matrix[1, 1] = fy
    camera_matrix[1, 2] = cy
    camera_matrix[0, 2] = cx
    mtx = camera_matrix
    dist = np.array([0.65208034, 12.0347837, -0.01159246, -0.0145476, -3052.98708447])
    mtx = np.array([[2708.31822406,     0.,  1170.96635697],
                    [0.,  2715.66754095,  1631.47033],
                    [0.,     0.,     1.]])

    pattern_contours = contour_image(pattern)
    pattern_centers = find_concentric(pattern_contours).astype(np.float32)

    # Find width of pattern (I am assuming the the width
    # from edge to edge of the pattern is 88mm, not the
    # width of the whole image)
    where_zero = np.where(pattern == 0)[0]
    pattern_size = where_zero[-1] - where_zero[0] + 1
    print_size = 88

    thresh = thresh_image(img)
    contours = contour_image(thresh)
    img_centers = find_concentric(contours, fuzzy=False)
    if len(img_centers) != 4:
        img_centers = find_concentric(contours, fuzzy=True)
        if len(img_centers) != 4:
            # Could maybe iterate through all subarrays until match is found (if len>4)
            raise Exception('Unable to find centers in picture')

    centers, _ = match_center_order(pattern_centers, img_centers, pattern, img)
    euler_vec, tvec = calc_params(
        pattern_centers / pattern_size * print_size, centers, mtx, False, imshape[::-1])
    roll, pitch, yaw = euler_vec
    # yaw = -yaw

    tx, ty, tz = tvec.reshape(-1)
    print(
        f'Camera Angles:\nRoll: {roll:.1f}°, Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°\nCamera Position:\n(x,y,z) = ({tx:.0f}, {ty:.0f}, {tz:.0f}) mm')
    return {'roll': float(roll),
            'pitch': float(pitch),
            'yaw': float(yaw),
            'tx': float(tx),
            'ty': float(ty),
            'tz': float(tz)}


if __name__ == '__main__':
    # could add argument to generate vis.html file from params.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", '--image', help="Location of input image", required=True)
    parser.add_argument("-p", '--pattern', help="Location of pattern image",
                        default='Camera Localization/pattern.png')
    args = parser.parse_args()
    main(args.image, args.pattern)
