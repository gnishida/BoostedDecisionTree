#include "ECP.h"
#include <QDir>
#include <iostream>

namespace ecp {

	cv::Vec3b convertLabelToColor(int label) {
		if (label == LABEL_WALL) {
			return cv::Vec3b(0, 255, 255);
		}
		else if (label == LABEL_WINDOW) {
			return cv::Vec3b(0, 0, 255);
		}
		else if (label == LABEL_DOOR) {
			return cv::Vec3b(0, 128, 255);
		}
		else if (label == LABEL_BALCONY) {
			return cv::Vec3b(255, 0, 128);
		}
		else if (label == LABEL_SHOP) {
			return cv::Vec3b(0, 255, 0);
		}
		else if (label == LABEL_ROOF) {
			return cv::Vec3b(255, 0, 0);
		}
		else if (label == LABEL_SKY) {
			return cv::Vec3b(255, 255, 128);
		}
		else {
			//return cv::Vec3b(0, 0, 0);
			// HACK
			// if the label is unknown, assume it is wall.
			return cv::Vec3b(0, 255, 255);
		}
	}

	int convertColorToLabel(const cv::Vec3b& color) {
		if (color == cv::Vec3b(0, 255, 255)) {
			return LABEL_WALL;
		}
		else if (color == cv::Vec3b(0, 0, 255)) {
			return LABEL_WINDOW;
		}
		else if (color == cv::Vec3b(0, 128, 255)) {
			return LABEL_DOOR;
		}
		else if (color == cv::Vec3b(255, 0, 128)) {
			return LABEL_BALCONY;
		}
		else if (color == cv::Vec3b(0, 255, 0)) {
			return LABEL_SHOP;
		}
		else if (color == cv::Vec3b(255, 0, 0)) {
			return LABEL_ROOF;
		}
		else if (color == cv::Vec3b(255, 255, 128)) {
			return LABEL_SKY;
		}
		else {
			return LABEL_UNKNOWN;
		}
	}

	void extractExampleFromPatch(const cv::Mat& patch, cv::Mat& feature) {
		feature = cv::Mat(1, patch.rows * patch.cols, CV_32F);

		for (int y = 0; y < patch.rows; ++y) {
			for (int x = 0; x < patch.cols; ++x) {
				cv::Vec3b col = patch.at<cv::Vec3b>(y, x);

				float val = ((float)col[0] + (float)col[1] + (float)col[2]) / 3.0f / 25.6;
				if (val >= 10) val = 9;
				if (val < 0) val = 0;

				feature.at<float>(y * patch.cols + x) = val;
			}
		}
	}

	void loadData(const QString& image_dir, const QString& ground_truth_image_dir, cv::Mat& X, cv::Mat& Y) {
		// create dataset
		//QDir ground_truth_dir(ground_truth_image_dir_path);
		//QDir images_dir(image_dir_path);

		QStringList image_files = QDir(image_dir).entryList(QDir::NoDotAndDotDot | QDir::Files);

		int patch_size = 2;
		int feature_size = patch_size * patch_size;

		// count the number of samples
		int num_samples = 0;
		for (int i = 0; i < image_files.size(); ++i) {
			cv::Mat image = cv::imread((image_dir + "/" + image_files[i]).toUtf8().constData());
			num_samples += (image.rows - patch_size + 1) * (image.cols - patch_size + 1);
		}

		// generate examples from the images
		X = cv::Mat(num_samples, feature_size, CV_32F);
		Y = cv::Mat(num_samples, 1, CV_32S);
		printf("Image processing:");
		int cnt = 0;
		for (int i = 0; i < image_files.size(); ++i) {
			printf("\rImage processing: %d", i + 1);

			// remove the file extension
			int index = image_files[i].lastIndexOf(".");
			QString filename = image_files[i].left(index);

			//std::cout << image_file.toUtf8().constData() << std::endl;
			cv::Mat image = cv::imread((image_dir + "/" + filename + ".jpg").toUtf8().constData());
			cv::Mat ground_truth = cv::imread((ground_truth_image_dir + "/" + filename + ".png").toUtf8().constData());
			//std::cout << "(" << image.rows << " x " << image.cols << ")" << std::endl;

			for (int y = patch_size; y < image.rows - patch_size + 1; y++) {
				for (int x = patch_size; x < image.cols - patch_size + 1; x++) {
					cv::Mat image_roi = image(cv::Rect(x, y, patch_size, patch_size));
					//std::cout << roi.rows << "," << roi.cols << std::endl;
					cv::Vec3b ground_truth_color = ground_truth.at<cv::Vec3b>(y + (patch_size - 1) / 2, x + (patch_size - 1) / 2);
					int label = convertColorToLabel(ground_truth_color);

					cv::Mat feature;
					extractExampleFromPatch(image_roi, feature);
					for (int k = 0; k < feature.cols; ++k) {
						X.at<float>(cnt, k) = feature.at<float>(0, k);
					}

					Y.at<int>(cnt, 0) = label;
					cnt++;
				}
			}
		}
		printf("\n");

		std::cout << "Dataset has been generated." << std::endl;
		std::cout << "#samples: " << X.rows << std::endl;
		std::cout << "#features: " << X.cols << std::endl;
	}

}