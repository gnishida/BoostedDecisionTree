#pragma once

#include <QString>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace ecp {

	static enum { LABEL_WALL = 0, LABEL_WINDOW, LABEL_DOOR, LABEL_BALCONY, LABEL_SHOP, LABEL_ROOF, LABEL_SKY, LABEL_UNKNOWN };

	cv::Vec3b convertLabelToColor(float label);
	float convertColorToLabel(const cv::Vec3b& color);
	void extractExampleFromPatch(const cv::Mat& patch, cv::Mat& feature);
	void extractFeaturesFromImage(const cv::Mat& image, cv::Mat& features);
	void extractFeaturesFromImage2(const cv::Mat& image, cv::Mat& features);
	void extractLabelsFromImage(const cv::Mat& image, cv::Mat& labels);

	void loadData(const QString& image_dir, const QString& ground_truth_image_dir, cv::Mat& X, cv::Mat& Y);

}