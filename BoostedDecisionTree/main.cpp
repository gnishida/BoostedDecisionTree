#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/ml/ml.hpp>

int main() {
	const int NUMBER_OF_CLASSES = 2;
	const int NUMBER_OF_TRAINING_SAMPLES = 6;
	const int NUMBER_OF_TEST_SAMPLES = 4;
	const int ATTRIBUTES_PER_SAMPLE = 3;

	// training/set data
	cv::Mat_<float> training_data(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE);
	cv::Mat_<int> training_classifications(NUMBER_OF_TRAINING_SAMPLES, 1);
	cv::Mat_<float> test_data(NUMBER_OF_TEST_SAMPLES, ATTRIBUTES_PER_SAMPLE);
	cv::Mat_<int> test_classifications(NUMBER_OF_TEST_SAMPLES, 1);

	// set training/test data
	training_data(0, 0) = 0;
	training_data(0, 1) = 0;
	training_data(0, 2) = 0;
	training_classifications(0, 0) = 0;
	training_data(1, 0) = 0;
	training_data(1, 1) = 0;
	training_data(1, 2) = 1;
	training_classifications(1, 0) = 0;
	training_data(2, 0) = 1;
	training_data(2, 1) = 1;
	training_data(2, 2) = 1;
	training_classifications(2, 0) = 1;
	training_data(3, 0) = 1;
	training_data(3, 1) = 0;
	training_data(3, 2) = 1;
	training_classifications(3, 0) = 1;
	training_data(4, 0) = 1;
	training_data(4, 1) = 0;
	training_data(4, 2) = 0;
	training_classifications(4, 0) = 0;
	training_data(5, 0) = 0;
	training_data(5, 1) = 1;
	training_data(5, 2) = 0;
	training_classifications(5, 0) = 1;
	test_data(0, 0) = 0;
	test_data(0, 1) = 0;
	test_data(0, 2) = 0;
	test_classifications(0, 0) = 0;
	test_data(1, 0) = 0;
	test_data(1, 1) = 0;
	test_data(1, 2) = 1;
	test_classifications(1, 0) = 0;
	test_data(2, 0) = 1;
	test_data(2, 1) = 1;
	test_data(2, 2) = 1;
	test_classifications(2, 0) = 1;
	test_data(3, 0) = 1;
	test_data(3, 1) = 0;
	test_data(3, 2) = 1;
	test_classifications(3, 0) = 1;

	
	cv::Ptr<cv::ml::Boost> boost = cv::ml::Boost::create();


	// set boost parameters
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


	boost->train(training_data, cv::ml::ROW_SAMPLE, training_classifications);
	boost->save("trained.xml");
	std::cout << "training done." << std::endl;

	for (int i = 0; i < NUMBER_OF_TEST_SAMPLES; ++i) {
		cv::Mat result;
		float ret = boost->predict(test_data.row(i), result);
		//float ret = boost->predict(test_data.row(i), cv::noArray(), cv::ml::StatModel::RAW_OUTPUT);
		std::cout << ret << std::endl;
		std::cout << result << std::endl;
	}

	return 0;
}