import cv2
import numpy as np
camera_matrix1 = np.array([[650.346616444449, 0, 0],
[0, 649.472545560264, 0],
[659.364563511604, 351.966016907267, 1]]).T
camera_matrix2 = np.array([[653.100835819782, 0, 0],
[0, 652.374708058560, 0],
[662.890788905584, 363.935569940028, 1]]).T
# (k1, k2, p1, p2, k3)
distCoeff1 = np.array([0.185736431470733, -0.250383574833521, 0, 0, 0.113275893049391])
distCoeff2 = np.array([0.189161563576057, -0.254669885642223, 0, 0, 0.115819997586013])
# 下方算式不用替换
u01 = camera_matrix1[0, 2]
u02 = camera_matrix2[0, 2]
dx_f = (1 / camera_matrix1[0, 0] + 1 / camera_matrix2[0, 0]) / 2
focus = (camera_matrix1[0, 0] + camera_matrix2[0, 0]) / 2
base = 60

def undistortion(img, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(img, camera_matrix, dist_coeff)
    return undistortion_image


def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

l1 = cv2.imread('ltt2.jpg')
cv_show('f', l1)
new_l1 = undistortion(l1, camera_matrix2, distCoeff2)
cv_show('after', new_l1)

r1 = cv2.imread('rtt2.jpg')
new_r1 = undistortion(r1, camera_matrix1, distCoeff1)
cv_show('af', new_r1)