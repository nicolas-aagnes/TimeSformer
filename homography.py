import numpy as np
import cv2
import skimage as io
import os
import random
import shutil

# No Occlusion
H_LC_1 =  np.array([[-7.45570216e+00,  2.50052903e-01,  1.22998523e+04],
                    [-1.59886298e+00, -7.65049176e+00,  4.97236885e+03],
                    [-3.07148730e-03,  2.99949796e-04,  1.00000000e+00]])
H_CL_1 =  np.array([[ 3.97156887e-02, -1.49414012e-02, -4.14202948e+02],
                    [ 5.94031292e-02, -1.31733814e-01, -7.56206025e+01],
                    [ 1.04168277e-04, -6.37879319e-06, -2.49536708e-01]])
H_CR_1 =  np.array([[ 1.84669078e+00, -3.04335377e-02, -3.02779889e+03],
                    [ 6.91965454e-01,  1.57569186e+00, -4.03583077e+02],
                    [ 1.01035547e-03, -2.77860842e-05,  1.00000000e+00]])
H_RC_1 =  np.array([[ 2.00546478e-01,  1.46856979e-02,  6.13141303e+02],
                    [-1.40971341e-01,  6.28867785e-01, -1.73032474e+02],
                    [-2.06540274e-04,  2.63599790e-06,  3.75701433e-01]])

# Truck Occlusion
H_LC_2 =  np.array([[ 1.18659063e+01, -1.03581832e+00, -1.51728898e+04],
                    [ 2.65481365e+00,  1.12722687e+01, -7.32983168e+03],
                    [ 3.31518648e-03, -4.49825472e-05,  1.00000000e+00]])
H_CL_2 =  np.array([[ 1.50603313e-02,  2.36495722e-03,  2.45843485e+02],
                    [-3.70978096e-02,  8.55607906e-02,  6.42652164e+01],
                    [-5.15965607e-05, -3.99153192e-06,  1.87873813e-01]])
H_CR_2 =  np.array([[ 1.41878778e+00,  7.22723014e-03, -1.56081044e+03],
                    [ 3.71992232e-01,  1.16845780e+00, -1.27144481e+02],
                    [ 6.61402508e-04,  1.14502912e-04,  1.00000000e+00]])
H_RC_2 =  np.array([[ 4.20269624e-01, -6.60572983e-02,  6.47562397e+02],
                    [-1.62025721e-01,  8.70763894e-01, -1.42178613e+02],
                    [-2.59414966e-04, -5.60145387e-05,  5.87980472e-01]])

white_truck = ["downtown-palo-alto-6", "dt-palo-alto-3", "mountain-view-2", "dt-san-jose",
               "sf-soma-1", "dt-san-jose-2", "downtown-palo-alto-1", "mountain-view-1",
               "sf-soma-2", "downtown-palo-alto-2", "dt-san-jose-3", "mountain-view-4",
               "sf-financial-4", "dt-palo-alto-2", "mountain-view-3", "sf-southbound-5",
               "dt-san-jose-4", "dt-palo-alto-1", "downtown-palo-alto-4", "downtown-palo-alto-6"]

no_truck = ["0927-2017-downtown-ann-1", "0928-2017-downtown-ann-1", "0927-2017-downtown-ann-3",
            "0927-2017-downtown-ann-2", "ANN-hanh-1", "ANN-hanh-2", "ANN-conor-2", "ANN-conor-1"]

data_path = '/vision/u/ajarno/STIP_dataset'
left_path = data_path + '/{:s}/L_camera'
right_path = data_path + '/{:s}/R_camera'
center_path = data_path + '/{:s}/C_camera'
img_path = '/{:06d}.jpg'
output_path = "/vision2/u/naagnes/github/TimeSformer/tmp_output"

def compareForAllVideos():
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    for f in sorted(os.listdir(data_path)):
        try:
            print("Video: ", f)
            # Selects random set of corresponding frames from timestamp data
            ts = np.genfromtxt(os.path.join(data_path, f, "timestamp_sync.txt"), delimiter=' ')
            selected_ts = ts[random.randint(0, len(ts) - 1)][2:]

            left = left_path.format(f) + img_path.format(int(selected_ts[0]))
            center = center_path.format(f) + img_path.format(int(selected_ts[1]))
            right = right_path.format(f) + img_path.format(int(selected_ts[2]))
            
            compareForOneWithResize(f, left, center, right)
        except:
            pass

def compareForOneWithResize(video_name, left_img_path, center_img_path, right_img_path, resize=(224, 224)):
    shape = cv2.imread(left_img_path).shape
    left = cv2.resize(cv2.imread(left_img_path), resize)
    center = cv2.resize(cv2.imread(center_img_path), resize)
    right = cv2.resize(cv2.imread(right_img_path), resize)
    
    S = np.array([[resize[0]/shape[1], 0, 0],[0, resize[1]/shape[0], 0],[0, 0, 1]])
    S_inv = np.linalg.inv(S)

    if video_name not in white_truck:
        im_ltc = cv2.warpPerspective(left, S @ H_LC_1 @ S_inv, (center.shape[1], center.shape[0]))
        im_rtc = cv2.warpPerspective(right, S @ H_RC_1 @ S_inv, (center.shape[1], center.shape[0]))
    else:
        im_ltc = cv2.warpPerspective(left, S @ H_LC_2 @ S_inv, (center.shape[1], center.shape[0]))
        im_rtc = cv2.warpPerspective(right, S @ H_RC_2 @ S_inv, (center.shape[1], center.shape[0]))

    # Can't display images in cluster
    #cv2.imshow("Destination Image", center)
    #cv2.imshow("Warped Source Left to Center", im_ltc)
    #cv2.imshow("Warped Source Right to Center", im_rtc)

    # Press 0 to go to the next set of images
    #cv2.waitKey(0)
    os.makedirs(os.path.join(output_path, video_name))

    cv2.imwrite(os.path.join(output_path, video_name, "center.png"), center)
    cv2.imwrite(os.path.join(output_path, video_name, "left_warp.png"), im_ltc)
    cv2.imwrite(os.path.join(output_path, video_name, "right_warp.png"), im_rtc)

if __name__ == "__main__":
    compareForAllVideos()