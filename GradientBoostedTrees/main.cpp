#include <opencv2/core.hpp>
#include <opencv2/ml/ml.hpp>

void genereateDataset(cv::Mat_<float>& X, cv::Mat_<float>& Y) {
	for (int i = 0; i < X.rows; ++i) {
		X(i, 0) = (float)(rand() % 1000) / 1000;
		X(i, 1) = (float)(rand() % 1000) / 1000;
		X(i, 2) = (float)(rand() % 1000) / 1000;

		float d = sqrtf((X(i, 0) - 0.5f) * (X(i, 0) - 0.5f) + (X(i, 1) - 0.5f) * (X(i, 1) - 0.5f) + (X(i, 2) - 0.5f) * (X(i, 2) - 0.5f));
		if (d <= 0.4) {
			Y(i, 0) = 1;
		}
		else {
			Y(i, 0) = 0;
		}
	}
}

void genereateDataset2(cv::Mat_<float>& X, cv::Mat_<float>& Y) {
	for (int i = 0; i < X.rows; ++i) {
		X(i, 0) = (float)(rand() % 1000) / 1000;
		X(i, 1) = (float)(rand() % 1000) / 1000;
		X(i, 2) = (float)(rand() % 1000) / 1000;

		if (X(i, 0) > 0.7 || (X(i, 1) > 0.7 && X(i, 2) > 0.7)) Y(i, 0) = 0;
		else if (X(i, 0) + X(i, 1) + X(i, 2) < 1) Y(i, 0) = 1;
		else Y(i, 0) = 2;
	}
}

int main() {
	// training dataset
	cv::Mat_<float> X_train(200, 3);
	cv::Mat_<float> Y_train(200, 1);
	genereateDataset2(X_train, Y_train);
	cv::Mat_<float> X_test(100, 3);
	cv::Mat_<float> Y_test(100, 1);
	genereateDataset2(X_test, Y_test);

	// Gradient boosted trees
	CvGBTreesParams params;
	params.loss_function_type = CvGBTrees::DEVIANCE_LOSS;
	params.weak_count = 100;
	params.max_depth = 10;
	params.subsample_portion = 1.0;

	CvGBTrees trees;
	trees.train(X_train, CV_ROW_SAMPLE, Y_train, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), params);

	// Test
	int correct = 0;
	for (int i = 0; i < Y_test.rows; ++i) {
		float pred = trees.predict(X_test.row(i));
		std::cout << pred << " - " << Y_test(i, 0) << std::endl;
		if (pred == Y_test(i, 0)) {
			correct++;
		}
	}
	std::cout << "Correct: " << correct << ", Incorrect: " << Y_test.rows - correct << ", Accuracy: " << (float)(correct) / Y_test.rows << std::endl;

	return 0;
}
