#include "MainWindow.h"
#include <time.h>
#include <QDir>
#include <opencv2/opencv.hpp>

cv::Vec3b convertLabelToColor(unsigned char label) {
	if (label == 0) {
		return cv::Vec3b(0, 255, 255);
	}
	else if (label == 1) {
		return cv::Vec3b(0, 0, 255);
	}
	else if (label == 2) {
		return cv::Vec3b(0, 128, 255);
	}
	else if (label == 3) {
		return cv::Vec3b(255, 0, 128);
	}
	else if (label == 4) {
		return cv::Vec3b(0, 255, 0);
	}
	else if (label == 5) {
		return cv::Vec3b(255, 0, 0);
	}
	else if (label == 6) {
		return cv::Vec3b(255, 255, 128);
	}
	else {
		//return cv::Vec3b(0, 0, 0);
		// HACK
		// if the label is unknown, assume it is wall.
		return cv::Vec3b(0, 255, 255);
	}
}

unsigned char convertColorToLabel(const cv::Vec3b& color) {
	if (color == cv::Vec3b(0, 255, 255)) {
		return 0;
	}
	else if (color == cv::Vec3b(0, 0, 255)) {
		return 1;
	}
	else if (color == cv::Vec3b(0, 128, 255)) {
		return 2;
	}
	else if (color == cv::Vec3b(255, 0, 128)) {
		return 3;
	}
	else if (color == cv::Vec3b(0, 255, 0)) {
		return 4;
	}
	else if (color == cv::Vec3b(255, 0, 0)) {
		return 5;
	}
	else if (color == cv::Vec3b(255, 255, 128)) {
		return 6;
	}
	else {
		return 7;
	}
}

std::vector<unsigned char> extractExampleFromPatch(const cv::Mat& patch, const cv::Vec3b& ground_truth) {
	std::vector<unsigned char> vec;

	for (int index = 0; index < patch.rows * patch.cols; ++index) {
		int y = index / patch.cols;
		int x = index % patch.cols;

		cv::Vec3b col = patch.at<cv::Vec3b>(y, x);

		float val = ((float)col[0] + (float)col[1] + (float)col[2]) / 3.0f / 25.6;
		if (val >= 10) val = 9;
		if (val < 0) val = 0;

		vec.push_back(val);
	}

	vec.push_back(convertColorToLabel(ground_truth));

	return vec;
}

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
	ui.setupUi(this);

	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(close()));
	connect(ui.actionTrainByECP, SIGNAL(triggered()), this, SLOT(onTrainByECP()));
}

MainWindow::~MainWindow() {
}

void MainWindow::onTrainByECP() {
	const int patch_size = 15;
	const int T = 10;
	const float r = 0.5;
	const int max_depth = 18;
	const int NUMBER_OF_CLASSES = 7;

	// create dataset
	time_t start = clock();
	QDir ground_truth_dir("../ECP/ground_truth/");
	QDir images_dir("../ECP/images/");

	QStringList image_files = images_dir.entryList(QDir::NoDotAndDotDot | QDir::Files);// , QDir::DirsFirst);

	// split the images into train and test
	std::vector<unsigned int> indices = std::vector<unsigned int>(image_files.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::random_shuffle(indices.begin(), indices.end());
	QStringList train_image_files;
	QStringList test_image_files;
	for (int i = 0; i < indices.size(); ++i) {
		if (i < image_files.size() * 0.8) {
			train_image_files.push_back(image_files[i]);
		}
		else {
			test_image_files.push_back(image_files[i]);
		}
	}


	std::vector<std::vector<unsigned char>> data;

	printf("Image processing: ");
	for (int i = 0; i < train_image_files.size(); ++i) {
		printf("\rImage processing: %d", i + 1);

		// remove the file extension
		int index = train_image_files[i].lastIndexOf(".");
		QString filename = train_image_files[i].left(index);

		//std::cout << image_file.toUtf8().constData() << std::endl;
		cv::Mat image = cv::imread((images_dir.absolutePath() + "/" + filename + ".jpg").toUtf8().constData());
		cv::Mat ground_truth = cv::imread((ground_truth_dir.absolutePath() + "/" + filename + ".png").toUtf8().constData());
		//std::cout << "(" << image.rows << " x " << image.cols << ")" << std::endl;

		for (int y = 0; y < image.rows - patch_size + 1; y++) {
			for (int x = 0; x < image.cols - patch_size + 1; x++) {
				cv::Mat image_roi = image(cv::Rect(x, y, patch_size, patch_size));
				//std::cout << roi.rows << "," << roi.cols << std::endl;
				cv::Vec3b ground_truth_color = ground_truth.at<cv::Vec3b>(y + (patch_size - 1) / 2, x + (patch_size - 1) / 2);

				std::vector<unsigned char> vec = extractExampleFromPatch(image_roi, ground_truth_color);

				data.push_back(vec);
			}
		}
	}
	printf("\n");

	cv::Mat_<float> train_data(data.size() * NUMBER_OF_CLASSES, data[0].size());
	for (int i = 0; i < data.size(); ++i) {
		for (int c = 0; c < NUMBER_OF_CLASSES; ++c) {
			for (int j = 0; j < data[i].size() - 1; ++j) {
				train_data(i * NUMBER_OF_CLASSES + c, j) = data[i][j];
			}
			train_data(i * NUMBER_OF_CLASSES + c, data[0].size() - 1) = c;
		}
	}
	cv::Mat_<float> train_labels(data.size() * NUMBER_OF_CLASSES, 1);
	for (int i = 0; i < data.size(); ++i) {
		for (int c = 0; c < NUMBER_OF_CLASSES; ++c) {
			if (data[i].back() == c) {
				train_labels(i * NUMBER_OF_CLASSES + c, 0) = 1;
			}
			else {
				train_labels(i * NUMBER_OF_CLASSES + c, 0) = 0;
			}
		}
	}


	time_t end = clock();

	std::cout << "Dataset has been created." << std::endl;
	std::cout << "#examples: " << train_data.rows << std::endl;
	std::cout << "#attributes: " << train_data.cols << std::endl;
	std::cout << "Elapsed: " << (end - start) / CLOCKS_PER_SEC << " sec." << std::endl;


	// setup boost
	start = clock();
	cv::Ptr<cv::ml::Boost> boost = cv::ml::Boost::create();

	boost->setBoostType(cv::ml::Boost::REAL);
	//boost->setBoostType(cv::ml::Boost::DISCRETE);
	boost->setWeakCount(100);
	//boost->setWeakCount(1);
	boost->setWeightTrimRate(0.95);
	boost->setMaxDepth(18);
	boost->setUseSurrogates(false);
	//boost->setPriors(? ? );
	boost->setPriors(cv::Mat());

	boost->setMaxCategories(15);
	boost->setMinSampleCount(1);
	/*
	boost->setCVFolds(5);
	boost->setUse1SERule(false);
	boost->setTruncatePrunedTree(false);
	*/
	boost->setRegressionAccuracy(0.0);

	// create random forest
	boost->train(train_data, cv::ml::ROW_SAMPLE, train_labels);
	//boost->save("trained.xml");
	end = clock();
	std::cout << "Random forest has been created." << std::endl;
	std::cout << "Elapsed: " << (end - start) / CLOCKS_PER_SEC << " sec." << std::endl;


	// test
	start = clock();
	QDir result_dir("results/");
	cv::Mat confusionMatrix(7, 7, CV_32F, cv::Scalar(0.0f));
	printf("Testing: ");
	for (int i = 0; i < test_image_files.size(); ++i) {
		printf("\rTesting: %d", i + 1);

		// remove the file extension
		int index = test_image_files[i].lastIndexOf(".");
		QString filename = test_image_files[i].left(index);

		cv::Mat image = cv::imread((images_dir.absolutePath() + "/" + filename + ".jpg").toUtf8().constData());
		cv::Mat ground_truth = cv::imread((ground_truth_dir.absolutePath() + "/" + filename + ".png").toUtf8().constData());

		cv::Mat result(image.size(), image.type(), cv::Vec3b(0, 0, 0));
		for (int y = 0; y < image.rows - patch_size + 1; y++) {
			for (int x = 0; x < image.cols - patch_size + 1; x++) {
				cv::Mat image_roi = image(cv::Rect(x, y, patch_size, patch_size));
				cv::Vec3b ground_truth_color = ground_truth.at<cv::Vec3b>(y + (patch_size - 1) / 2, x + (patch_size - 1) / 2);
				unsigned char ground_truth_label = convertColorToLabel(ground_truth_color);

				float max_ret = 0.0f;
				int best_c = -1;
				for (int c = 0; c < NUMBER_OF_CLASSES; ++c) {
					std::vector<unsigned char> vec = extractExampleFromPatch(image_roi, cv::Vec3b(0, 0, 0));
					vec.back() = c;
					cv::Mat_<float> test_data(1, vec.size());
					for (int k = 0; k < vec.size() - 1; ++k) {
						test_data(0, k) = vec[k];
					}

					cv::Mat result;
					float ret = boost->predict(test_data.row(i), result);

					if (ret > max_ret) {
						max_ret = ret;
						best_c = c;
					}
				}

				result.at<cv::Vec3b>(y + (patch_size - 1) / 2, x + (patch_size - 1) / 2) = convertLabelToColor(best_c);

				// update confusion matrix
				confusionMatrix.at<float>(ground_truth_label, best_c) += 1;
			}
		}

		cv::imwrite((result_dir.absolutePath() + "/" + filename + ".png").toUtf8().constData(), result);
	}
	printf("\n");
	end = clock();
	std::cout << "Test has been finished." << std::endl;
	std::cout << "Elapsed: " << (end - start) / CLOCKS_PER_SEC << " sec." << std::endl;

	std::cout << "Confusion matrix:" << std::endl;
	cv::Mat confusionMatrixSum;
	cv::reduce(confusionMatrix, confusionMatrixSum, 1, cv::REDUCE_SUM);
	for (int r = 0; r < confusionMatrix.rows; ++r) {
		for (int c = 0; c < confusionMatrix.cols; ++c) {
			if (c > 0) std::cout << ", ";
			std::cout << confusionMatrix.at<float>(r, c) / confusionMatrixSum.at<float>(r, 0);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

}