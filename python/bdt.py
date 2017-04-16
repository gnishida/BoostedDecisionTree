from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2
from scipy import misc
from skimage import io, color
import os
import numpy as np
import random

def convertColorToLable(color):
	if color[0] == 255 and color[1] == 255 and color[2] == 0:
		return 0
	elif color[0] == 255 and color[1] == 0 and color[2] == 0:
		return 1
	elif color[0] == 255 and color[1] == 128 and color[2] == 0:
		return 2
	elif color[0] == 128 and color[1] == 0 and color[2] == 255:
		return 3
	elif color[0] == 0 and color[1] == 255 and color[2] == 0:
		return 4
	elif color[0] == 0 and color[1] == 0 and color[2] == 255:
		return 5
	elif color[0] == 128 and color[1] == 255 and color[2] == 255:
		return 6
	else:
		return 7
	
def extractFeatureFromPatch(patch):
	feature = []
	for y in xrange(patch.shape[0]):
		for x in xrange(patch.shape[1]):
			val = float(patch[y,x]) / 25.6
			if val >= 10: val = 9
			if val < 0: val = 0
			
			feature.append(val)
	return feature
	

def generateDataset(images_dir, ground_truth_dir, files, PATCH_SIZE):
	X = []
	Y = []
	for file in files:
		filename = os.path.splitext(os.path.basename(file))[0]
		#print filename
		
		image = np.array(io.imread(images_dir + filename + ".jpg"))
		image = color.rgb2lab(image)
		image = image[:,:,0]
		
		ground_truth = np.array(io.imread(ground_truth_dir + filename + ".png"))
		
		
		
		for y in xrange(PATCH_SIZE, image.shape[0] - PATCH_SIZE):
			for x in xrange(PATCH_SIZE, image.shape[1] - PATCH_SIZE):
				patch = image[y-PATCH_SIZE:y+PATCH_SIZE+1,x-PATCH_SIZE:x+PATCH_SIZE+1]
				feature = extractFeatureFromPatch(patch)
				for f in feature:
					X.append(f)
				
				label = convertColorToLable(ground_truth[y,x])
				Y.append(label)

	num_cols = (PATCH_SIZE*2+1) ** 2
	X = np.reshape(X, newshape=(len(X)/num_cols, num_cols))
	Y = np.reshape(Y, newshape=(len(Y), 1))
	
	return X, Y

def main():
	images_dir = "../ECP/images/"
	ground_truth_dir = "../ECP/ground_truth/"
	files = os.listdir(images_dir)

	files_train = []
	files_val = []
	files_test = []
	random.shuffle(files)
	for i in xrange(len(files)):
		if i < float(len(files)) * 0.8:
			files_train.append(files[i])
		elif i < float(len(files)) * 0.8:
			files_val.append(files[i])
		else:
			files_test.append(files[i])

	PATCH_SIZE = 7
	X_train, y_train = generateDataset(images_dir, ground_truth_dir, files_train, PATCH_SIZE)
	#X_val, y_val = generateDataset(images_dir, ground_truth_dir, files_val, PATCH_SIZE)
	X_test, y_test = generateDataset(images_dir, ground_truth_dir, files_test, PATCH_SIZE)


	print "----------------------------------------"
	print "Features have been extracted."
	print "Training dataset:"
	print X_train.shape
	print y_train.shape
	print "Test dataset:"
	print X_test.shape
	print y_test.shape
	print "----------------------------------------"

	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=10, random_state=0).fit(X_train, y_train)
	print clf.score(X_test, y_test)

if __name__ == "__main__":
    main()