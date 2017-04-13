#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/ml/ml.hpp>

int main() {
	const int NUMBER_OF_CLASSES = 2;
	const int NUMBER_OF_TRAINING_SAMPLES = 15;
	const int NUMBER_OF_TEST_SAMPLES = 1;
	const int ATTRIBUTES_PER_SAMPLE = 6;

	// training/set data
	cv::Mat_<float> training_data(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE);
	cv::Mat_<int> training_classifications(NUMBER_OF_TRAINING_SAMPLES, 1);
	cv::Mat_<float> test_data(NUMBER_OF_TEST_SAMPLES, ATTRIBUTES_PER_SAMPLE);
	cv::Mat_<int> test_classifications(NUMBER_OF_TEST_SAMPLES, 1);

	// set training/test data
	training_data(0, 0) = 0;
	training_data(0, 1) = 0;
	training_data(0, 2) = 0;
	training_data(0, 3) = 0;
	training_data(0, 4) = 0;
	training_data(0, 5) = 2;
	training_classifications(0, 0) = 1;
	training_data(1, 0) = 2;
	training_data(1, 1) = 0;
	training_data(1, 2) = 0;
	training_data(1, 3) = 0;
	training_data(1, 4) = 0;
	training_data(1, 5) = 1;
	training_classifications(1, 0) = 0;
	training_data(2, 0) = 1;
	training_data(2, 1) = 2;
	training_data(2, 2) = 0;
	training_data(2, 3) = 0;
	training_data(2, 4) = 0;
	training_data(2, 5) = 1;
	training_classifications(2, 0) = 0;
	training_data(3, 0) = 1;
	training_data(3, 1) = 1;
	training_data(3, 2) = 0;
	training_data(3, 3) = 0;
	training_data(3, 4) = 0;
	training_data(3, 5) = 1;
	training_classifications(3, 0) = 0;
	training_data(4, 0) = 1;
	training_data(4, 1) = 3;
	training_data(4, 2) = 2;
	training_data(4, 3) = 2;
	training_data(4, 4) = 0;
	training_data(4, 5) = 1;
	training_classifications(4, 0) = 0;
	training_data(5, 0) = 0;
	training_data(5, 1) = 0;
	training_data(5, 2) = 0;
	training_data(5, 3) = 0;
	training_data(5, 4) = 4;
	training_data(5, 5) = 1;
	training_classifications(5, 0) = 0;
	training_data(6, 0) = 1;
	training_data(6, 1) = 4;
	training_data(6, 2) = 0;
	training_data(6, 3) = 0;
	training_data(6, 4) = 1;
	training_data(6, 5) = 1;
	training_classifications(6, 0) = 1;
	training_data(7, 0) = 1;
	training_data(7, 1) = 4;
	training_data(7, 2) = 0;
	training_data(7, 3) = 0;
	training_data(7, 4) = 2;
	training_data(7, 5) = 1;
	training_classifications(7, 0) = 1;
	training_data(8, 0) = 1;
	training_data(8, 1) = 4;
	training_data(8, 2) = 0;
	training_data(8, 3) = 0;
	training_data(8, 4) = 3;
	training_data(8, 5) = 1;
	training_classifications(8, 0) = 1;
	training_data(9, 0) = 1;
	training_data(9, 1) = 3;
	training_data(9, 2) = 1;
	training_data(9, 3) = 1;
	training_data(9, 4) = 1;
	training_data(9, 5) = 1;
	training_classifications(9, 0) = 1;
	training_data(10, 0) = 1;
	training_data(10, 1) = 3;
	training_data(10, 2) = 1;
	training_data(10, 3) = 1;
	training_data(10, 4) = 2;
	training_data(10, 5) = 1;
	training_classifications(10, 0) = 1;
	training_data(11, 0) = 1;
	training_data(11, 1) = 3;
	training_data(11, 2) = 1;
	training_data(11, 3) = 2;
	training_data(11, 4) = 1;
	training_data(11, 5) = 1;
	training_classifications(11, 0) = 1;
	training_data(12, 0) = 1;
	training_data(12, 1) = 3;
	training_data(12, 2) = 1;
	training_data(12, 3) = 2;
	training_data(12, 4) = 2;
	training_data(12, 5) = 1;
	training_classifications(12, 0) = 1;
	training_data(13, 0) = 1;
	training_data(13, 1) = 3;
	training_data(13, 2) = 1;
	training_data(13, 3) = 1;
	training_data(13, 4) = 3;
	training_data(13, 5) = 1;
	training_classifications(13, 0) = 0;
	training_data(14, 0) = 1;
	training_data(14, 1) = 3;
	training_data(14, 2) = 1;
	training_data(14, 3) = 2;
	training_data(14, 4) = 3;
	training_data(14, 5) = 1;
	training_classifications(14, 0) = 1;
	test_data(0, 0) = 2;
	test_data(0, 1) = 3;
	test_data(0, 2) = 2;
	test_data(0, 3) = 1;
	test_data(0, 4) = 2;
	test_data(0, 5) = 1;
	test_classifications(0, 0) = 0;

	std::cout << training_data << std::endl;
	std::cout << training_classifications << std::endl;
	std::cout << test_data << std::endl;
	std::cout << test_classifications << std::endl;


	// unroll the data
	cv::Mat_<float> new_data(NUMBER_OF_TRAINING_SAMPLES * NUMBER_OF_CLASSES, ATTRIBUTES_PER_SAMPLE + 1);
	cv::Mat_<int> new_responses(NUMBER_OF_TRAINING_SAMPLES * NUMBER_OF_CLASSES, 1);
	for (int i = 0; i < NUMBER_OF_TRAINING_SAMPLES; ++i) {
		for (int j = 0; j < NUMBER_OF_CLASSES; ++j) {
			for (int k = 0; k < ATTRIBUTES_PER_SAMPLE; ++k) {
				new_data((i * NUMBER_OF_CLASSES) + j, k) = training_data(i, k);
			}
			new_data((i * NUMBER_OF_CLASSES) + j, ATTRIBUTES_PER_SAMPLE) = (float)j;

			if ((int)training_classifications(i, 0) == j) {
				new_responses((i * NUMBER_OF_CLASSES) + j, 0) = 1;
			}
			else {
				new_responses((i * NUMBER_OF_CLASSES) + j, 0) = 0;
			}
		}
	}

	std::cout << new_data << std::endl;
	std::cout << new_responses << std::endl;


	
	// define attributes
	// +1 for the multiple classes
	// another +1 for label
	cv::Mat var_type = cv::Mat(ATTRIBUTES_PER_SAMPLE + 2, 1, CV_8U);
	for (int i = 0; i < var_type.rows; ++i) {
		if (i < ATTRIBUTES_PER_SAMPLE) {
			var_type.at<uchar>(i, 0) = cv::ml::VAR_NUMERICAL;
		}
		else {
			var_type.at<uchar>(i, 0) = cv::ml::VAR_CATEGORICAL;
		}
	}

	std::cout << var_type << std::endl;


	cv::Ptr<cv::ml::Boost> boost = cv::ml::Boost::create();


	// set boost parameters
	//boost->setBoostType(cv::ml::Boost::REAL);
	boost->setBoostType(cv::ml::Boost::DISCRETE);
	boost->setWeakCount(100);
	boost->setWeightTrimRate(0.95);
	//boost->setMaxDepth(18);
	boost->setMaxDepth(2);
	boost->setUseSurrogates(false);
	//boost->setPriors(? ? );
	boost->setPriors(cv::Mat());

	/*
	boost->setMaxCategories(15);
	boost->setMinSampleCount(5);
	boost->setCVFolds(5);
	boost->setUse1SERule(false);
	boost->setTruncatePrunedTree(false);
	boost->setRegressionAccuracy(0.0);
	*/


	boost->train(new_data, cv::ml::ROW_SAMPLE, new_responses);
	std::cout << "training done." << std::endl;

	//cv::ml::Boost::Params params;
	//CvBoostParams params;

	for (int i = 0; i < NUMBER_OF_TEST_SAMPLES; ++i) {
		cv::Mat_<float> new_sample(1, ATTRIBUTES_PER_SAMPLE + 1);

		for (int k = 0; k < ATTRIBUTES_PER_SAMPLE; ++k) {
			new_sample(0, k) = test_data(0, k);
		}

		for (int c = 0; c < NUMBER_OF_CLASSES; ++c) {
			new_sample(0, ATTRIBUTES_PER_SAMPLE) = (float)c;

			cv::Mat result;
			boost->predict(new_sample, result);
			std::cout << result << std::endl;
		}
	}

	return 0;
}