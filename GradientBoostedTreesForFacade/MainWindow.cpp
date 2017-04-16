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
	params.weak_count = 100;
	params.max_depth = 18;
	params.subsample_portion = 0.3;

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

	cv::Mat_<float> confusionMat(7, 7, 0.0f);

	// test for each image
	printf("Image testing:");
	int correct = 0;
	int total = 0;
	for (int i = 0; i < image_files.size(); ++i) {
		printf("\rImage testing: %d", i + 1);

		// remove the file extension
		int index = image_files[i].lastIndexOf(".");
		QString filename = image_files[i].left(index);

		// load image
		cv::Mat image = cv::imread((test_image_dir + "/" + filename + ".jpg").toUtf8().constData());
		cv::Mat ground_truth = cv::imread((ground_truth_image_dir + "/" + filename + ".png").toUtf8().constData());

		// initialize result image
		cv::Mat result(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));

		// extract features
		cv::Mat X_test;
		cv::Mat Y_test;
		ecp::extractFeaturesFromImage(image, X_test);
		ecp::extractLabelsFromImage(ground_truth, Y_test);

		// estimate label for each pixel
		for (int y = 0; y < image.rows; y++) {
			for (int x = 0; x < image.cols; x++) {
				int idx = y * image.cols + x;

				// test
				float pred = trees.predict(X_test.row(idx));

				result.at<cv::Vec3b>(y, x) = ecp::convertLabelToColor(pred);

				// update confusion matrix
				float true_label = Y_test.at<float>(idx, 0);
				confusionMat((int)true_label, (int)pred) += 1;

				if (true_label == pred) correct++;
				total++;
			}
		}

		// save the result image
		cv::imwrite((result_dir + "/" + filename + ".png").toUtf8().constData(), result);
	}
	printf("\n");

	std::cout << "accuracy = " << (float)correct / total << " (" << correct << " / " << total << ")" << std::endl;

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