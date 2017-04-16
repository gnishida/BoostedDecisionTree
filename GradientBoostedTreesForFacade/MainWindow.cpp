#include "MainWindow.h"
#include <opencv2/core.hpp>
#include <opencv2/ml/ml.hpp>
#include "ECP.h"
#include <QDir>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
	ui.setupUi(this);

	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(close()));
	connect(ui.actionRun, SIGNAL(triggered()), this, SLOT(onRun()));
}

MainWindow::~MainWindow() {
}

void MainWindow::onRun() {
	QString train_image_dir = "../ECP/train_images/";
	QString test_image_dir = "../ECP/test_images/";
	QString ground_truth_image_dir = "../ECP/ground_truth/";
	QString result_dir = "results/";


	// training dataset
	cv::Mat X_train;
	cv::Mat Y_train;
	ecp::loadData(train_image_dir, ground_truth_image_dir, X_train, Y_train);


	// Gradient boosted trees
	CvGBTreesParams params;
	params.loss_function_type = CvGBTrees::DEVIANCE_LOSS;
	params.weak_count = 10;
	params.max_depth = 2;
	params.subsample_portion = 1.0;

	CvGBTrees trees;
	std::cout << "Training ...";
	trees.train(X_train, CV_ROW_SAMPLE, Y_train, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), params);
	std::cout << " done." << std::endl;


	// release memory for training dataset
	X_train = cv::Mat();
	Y_train = cv::Mat();


	//////////////////////////////////////////////////////////////////////////////////////////
	// test

	if (!QDir(result_dir).exists()) {
		QDir().mkdir(result_dir);
	}
	QStringList image_files = QDir(test_image_dir).entryList(QDir::NoDotAndDotDot | QDir::Files);

	int patch_size = 2;
	int feature_size = patch_size * patch_size;

	cv::Mat_<float> confusionMat(7, 7, 0.0f);

	// count the number of samples
	for (int i = 0; i < image_files.size(); ++i) {
		// remove the file extension
		int index = image_files[i].lastIndexOf(".");
		QString filename = image_files[i].left(index);

		cv::Mat image = cv::imread((test_image_dir + "/" + filename + ".jpg").toUtf8().constData());
		cv::Mat ground_truth = cv::imread((ground_truth_image_dir + "/" + filename + ".png").toUtf8().constData());

		cv::Mat result(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));

		for (int y = patch_size; y < image.rows - patch_size + 1; y++) {
			for (int x = patch_size; x < image.cols - patch_size + 1; x++) {
				cv::Mat image_roi = image(cv::Rect(x, y, patch_size, patch_size));
				//std::cout << roi.rows << "," << roi.cols << std::endl;
				cv::Vec3b ground_truth_color = ground_truth.at<cv::Vec3b>(y + (patch_size - 1) / 2, x + (patch_size - 1) / 2);
				int label = ecp::convertColorToLabel(ground_truth_color);

				cv::Mat X_test;
				ecp::extractExampleFromPatch(image_roi, X_test);

				// test
				float pred = trees.predict(X_test);

				result.at<cv::Vec3b>(y, x) = ecp::convertLabelToColor(pred);

				confusionMat(label, (int)pred) += 1;
			}
		}


		cv::imwrite((result_dir + "/" + filename + ".png").toUtf8().constData(), result);
	}


	// output confusion matrix
	cv::Mat_<float> confusionMatSum;
	cv::reduce(confusionMat, confusionMatSum, 1, CV_REDUCE_SUM);

	for (int r = 0; r < confusionMat.rows; ++r) {
		for (int c = 0; c < confusionMat.cols; ++c) {
			confusionMat(r, c) /= confusionMatSum(r, 0);
		}
	}

	std::cout << confusionMat << std::endl;

}