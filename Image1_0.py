import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the images to be aligned
img_list = []
# im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im1 = cv2.imread("img1.jpg",0)
im2 = cv2.imread("img2.jpg",0)
im3 = cv2.imread("img3.jpg",0)
im4 = cv2.imread("img4.jpg",0)
im5 = cv2.imread("img5.jpg",0)

img_list.append(im1)
img_list.append(im2)
img_list.append(im3)
img_list.append(im4)
img_list.append(im5)

# Convert images to grayscale
crop_list = []
for img in img_list:
	#crop
	y1 = 380
	y2 = 380+190
	x1 = 260
	x2 = x1+120
	# im1_gray = im1_gray[x1:x2, y1:y2]
	img = img[x1:x2, y1:y2]
	# cv2.imshow("image 1 cropped",im1_gray)
	# cv2.waitKey(0)
	crop_list.append(img)
	cv2.imshow("image cropped",img)
	cv2.waitKey(0)

#histogram

def hist(im1_gray):
	hist,bins = np.histogram(im1_gray.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max()/ cdf.max()
	# plt.plot(cdf_normalized, color = 'b')
	# plt.hist(im1_gray.flatten(),256,[0,256], color = 'r')
	# plt.xlim([0,256])
	# plt.legend(('cdf','histogram'), loc = 'upper left')
	# plt.show()
	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')
	img2 = cdf[im1_gray]
	cv2.imshow("Histogram Image",img2)
	cv2.waitKey(0)
	return img2

hist_list = []
for img in crop_list:
	img = hist(img)
	hist_list.append(img)
i=1
for img in hist_list:
    cv2.waitKey()
    file_name = "normalised" + str(i) + ".jpg"
    cv2.imwrite(file_name,img)
    i+=1
#radiometric

reg_list = []
reg_list.append(hist_list[0])
for img in hist_list:
	#registration
	# Find size of image1
	sz = hist_list[0].shape

	# Define the motion model
	warp_mode = cv2.MOTION_TRANSLATION

	# Define 2x3 or 3x3 matrices and initialize the matrix to identity
	if warp_mode == cv2.MOTION_HOMOGRAPHY :
		warp_matrix = np.eye(3, 3, dtype=np.float32)
	else :
		warp_matrix = np.eye(2, 3, dtype=np.float32)

	# Specify the number of iterations.
	number_of_iterations = 5000;

	# Specify the threshold of the increment
	# in the correlation coefficient between two iterations
	termination_eps = 1e-10;

	# Define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

	# Run the ECC algorithm. The results are stored in warp_matrix.
	(cc, warp_matrix) = cv2.findTransformECC (hist_list[0],img,warp_matrix, warp_mode, criteria)

	if warp_mode == cv2.MOTION_HOMOGRAPHY :
		# Use warpPerspective for Homography
		im2_aligned = cv2.warpPerspective (img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	else :
		# Use warpAffine for Translation, Euclidean and Affine
		im2_aligned = cv2.warpAffine(img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

	reg_list.append(im2_aligned);
	# Show final results
	cv2.imshow("Image 1", hist_list[0])
	cv2.imshow("Aligned Image 2", im2_aligned)
	cv2.waitKey(0)

i=1
for img in reg_list:
    cv2.waitKey()
    file_name = "registered" + str(i) + ".jpg"
    cv2.imwrite(file_name,img)
    i+=1

#SKELETONISATION
def skeleton(img):
	size = np.size(img)
	skel = np.zeros(img.shape,np.uint8)
	blur = cv2.GaussianBlur(img, (5, 5), 0)
	ret3, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done = False

	while not done:
		eroded = cv2.erode(img,element)
		temp = cv2.dilate(eroded,element)
		temp = cv2.subtract(img,temp)
		skel = cv2.bitwise_or(skel,temp)
		img = eroded.copy()

		zeros = size - cv2.countNonZero(img)
		if zeros==size:
			done = True

	cv2.imshow("skeleton",skel)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return skel

skel_list = []
for img in reg_list:
	img_1_skel = skeleton(img)
	skel_list.append(img_1_skel)
	#img_2_skel = skeleton(im2_aligned)
# cv2.imwrite("skel_1_old.png",img_1_skel)
#
# cv2.imwrite("skel_2_old.png",img_2_skel)
thinnig = []
for img in skel_list:
	img1_hitmiss = cv2.morphologyEx(img,cv2.MORPH_HITMISS,kernel=np.ones((3,3),np.uint8))
	#img2_hitmiss = cv2.morphologyEx(img_2_skel,cv2.MORPH_HITMISS,kernel=np.ones((3,3),np.uint8))
	img1_open = img-img1_hitmiss
	thinnig.append(img1_open)

	#img2_open = img_2_skel - img2_hitmiss
# img1_open = img_1_skel - cv2.morphologyEx(img_1_skel,cv2.MORPH_OPEN,kernel=np.ones((2,2),np.uint8))
# img2_open = img_2_skel - cv2.morphologyEx(img_2_skel,cv2.MORPH_OPEN,kernel=np.ones((2,2),np.uint8))
final_list = []
for img in thinnig:
	final_list.append(img[0:70,:])
# img2_open = img2_open[0:70,:]

final = "final"
i = 1
for img in final_list:
    cv2.imshow("smooth",img)
    cv2.waitKey()
    file_name = final + str(i) + ".jpg"
    cv2.imwrite(file_name,img)
    i+=1
#cv2.imwrite("smooth2.png",img2_open)

print('a')