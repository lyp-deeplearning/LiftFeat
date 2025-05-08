import os
import sys
import torch
import numpy as np
import math
import cv2

os.environ['CUDA_VISIBLE_DEVICES']='1'

from models.liftfeat_wrapper import LiftFeat,MODEL_PATH


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches


if __name__=="__main__":
    liftfeat=LiftFeat(weight=MODEL_PATH,detect_threshold=0.05)
    
    img1=cv2.imread(os.path.join(os.path.dirname(__file__),'./assert/ref.jpg'))
    img2=cv2.imread(os.path.join(os.path.dirname(__file__),'./assert/query.jpg'))
    
    # import pdb;pdb.set_trace()
    mkpts1,mkpts2=liftfeat.match_liftfeat(img1,img2)
    canvas=warp_corners_and_draw_matches(mkpts1,mkpts2,img1,img2)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=[12,12])
    plt.imshow(canvas[...,::-1])
    
    plt.savefig(os.path.join(os.path.dirname(__file__),'match.jpg'), dpi=300, bbox_inches='tight')
    
    plt.show()
    