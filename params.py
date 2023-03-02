import numpy as np

camera_matrix1 = np.array([[707.039703801139	,0	,0],
[0	,709.116786228983,	0],
[315.691985099822,242.724082140330,1]]).T
camera_matrix2 = np.array([[705.168670286673,	0,	0] ,
[0,706.044581265140,	0],
[297.537650660022,	254.939159553401,1]] ).T
# (k1, k2, p1, p2, k3)
distCoeff1 = np.array([-0.125664135896349,	0.565711277731587,0,0,	-0.725355828526992])
distCoeff2 = np.array([-0.148527135645977,	1.97568258641427,0,0,	-13.3874703447732])


# 下方算式不用替换
u01 = camera_matrix1[0, 2]
u02 = camera_matrix2[0, 2]
dx_f = (1 / camera_matrix1[0, 0] + 1 / camera_matrix2[0, 0]) / 2
focus = (camera_matrix1[0, 0] + camera_matrix2[0, 0]) / 2
base = 250
