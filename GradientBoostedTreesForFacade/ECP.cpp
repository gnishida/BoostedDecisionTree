#include "ECP.h"
#include <QDir>
#include <iostream>

namespace ecp {

	cv::Vec3b convertLabelToColor(float label) {
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

	float convertColorToLabel(const cv::Vec3b& color) {
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


	void extractFeaturesFromImage(const cv::Mat& image, cv::Mat& features) {
		cv::Mat image_lab;
		cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);
		image_lab.convertTo(image_lab, CV_32F);

		// split the lab image into 3 channels
		cv::Mat channels[3];
		cv::split(image_lab, channels);

		// apply Gaussian filter
		cv::Mat image_gauss[4][3];
		for (int i = 0; i < 4; ++i) {
			int sigma = std::pow(2, i);
			for (int ch = 0; ch < 3; ++ch) {
				cv::GaussianBlur(channels[ch], image_gauss[i][ch], cv::Size(0, 0), sigma, sigma, cv::BORDER_DEFAULT);
			}
		}

		// apply derivative of Gaussian filter
		cv::Mat image_soblex[2];
		cv::Mat image_sobley[2];
		for (int i = 0; i < 2; ++i) {
			cv::Sobel(image_gauss[i + 1][0], image_soblex[i], CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
			cv::Sobel(image_gauss[i + 1][0], image_sobley[i], CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
		}

		// apply LoG
		cv::Mat image_log[4];
		for (int i = 0; i < 4; ++i) {
			cv::Laplacian(image_gauss[i][0], image_log[i], CV_32F, 3, 1, 0, cv::BORDER_DEFAULT);
		}

		// for each pixel, get 17-dimensional feature
		features = cv::Mat(image.rows * image.cols, 17, CV_32F);
		for (int y = 0; y < image.rows; ++y) {
			for (int x = 0; x < image.cols; ++x) {
				int idx = y * image.cols + x;

				int cnt = 0;

				// Gaussian filter
				for (int i = 0; i < 3; ++i) {
					for (int ch = 0; ch < 3; ++ch) {
						features.at<float>(idx, cnt) = image_gauss[i][ch].at<float>(y, x);
						cnt++;
					}
				}

				// derivative of Gaussian
				for (int i = 0; i < 2; ++i) {
					features.at<float>(idx, cnt) = image_soblex[i].at<float>(y, x);
					cnt++;
					features.at<float>(idx, cnt) = image_sobley[i].at<float>(y, x);
					cnt++;
				}

				// LoG
				for (int i = 0; i < 4; ++i) {
					features.at<float>(idx, cnt) = image_log[i].at<float>(y, x);
					cnt++;
				}

				//std::cout << features.row(idx) << std::endl;
			}
		}
	}

	void extractFeaturesFromImage2(const cv::Mat& image, cv::Mat& features) {
		// for each pixel, get 5x5x3-dimensional feature
		features = cv::Mat(image.rows * image.cols, 75, CV_32F);
		for (int y = 0; y < image.rows; ++y) {
			for (int x = 0; x < image.cols; ++x) {
				int idx = y * image.cols + x;

				int cnt = 0;

				for (int yy = y - 2; yy <= y + 2; yy++) {
					for (int xx = x - 2; xx <= x + 2; xx++) {
						int u = xx;
						int v = yy;
						if (u < 0) u = 0;
						if (u >= image.cols) u = image.cols - 1;
						if (v < 0) v = 0;
						if (v >= image.rows) v = image.rows - 1;

						cv::Vec3b col = image.at<cv::Vec3b>(v, u);
						for (int i = 0; i < 3; ++i) {
							features.at<float>(idx, cnt++) = col[i];
						}
					}
				}
			}
		}
	}



	void extractLabelsFromImage(const cv::Mat& image, cv::Mat& labels) {
		// for each pixel, get label
		labels = cv::Mat(image.rows * image.cols, 1, CV_32F);
		for (int y = 0; y < image.rows; ++y) {
			for (int x = 0; x < image.cols; ++x) {
				float label = convertColorToLabel(image.at<cv::Vec3b>(y, x));
				labels.at<float>(y * image.cols + x, 0) = label;
			}
		}
	}

	void loadData(const QString& image_dir, const QString& ground_truth_image_dir, cv::Mat& X, cv::Mat& Y) {
		QStringList image_files = QDir(image_dir).entryList(QDir::NoDotAndDotDot | QDir::Files);

		const int feature_size = 17;

		// count the number of samples
		int num_samples = 0;
		for (int i = 0; i < image_files.size(); ++i) {
			cv::Mat image = cv::imread((image_dir + "/" + image_files[i]).toUtf8().constData());
			num_samples += image.rows * image.cols;
		}

		// generate examples from the images
		X = cv::Mat(num_samples, feature_size, CV_32F);
		Y = cv::Mat(num_samples, 1, CV_32F);
		printf("Image processing:");
		int cnt = 0;
		for (int i = 0; i < image_files.size(); ++i) {
			printf("\rImage processing: %d", i + 1);

			// remove the file extension
			int index = image_files[i].lastIndexOf(".");
			QString filename = image_files[i].left(index);

			cv::Mat image = cv::imread((image_dir + "/" + filename + ".jpg").toUtf8().constData());
			cv::Mat ground_truth = cv::imread((ground_truth_image_dir + "/" + filename + ".png").toUtf8().constData());

			// extract features from the image
			cv::Mat features;
			extractFeaturesFromImage(image, features);
			cv::Mat roi(X, cv::Rect(0, cnt, features.cols, features.rows));
			features.copyTo(roi);

			cv::Mat labels;
			extractLabelsFromImage(ground_truth, labels);
			//std::cout << labels << std::endl;
			cv::Mat roi2(Y, cv::Rect(0, cnt, labels.cols, labels.rows));
			labels.copyTo(roi2);

			cnt += features.rows;
		}
		printf("\n");

		std::cout << "Dataset has been generated." << std::endl;
		std::cout << "#samples: " << X.rows << std::endl;
		std::cout << "#features: " << X.cols << std::endl;
	}

}