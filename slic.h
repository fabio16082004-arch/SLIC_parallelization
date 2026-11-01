//
// Created by fbi168 on 25/10/26.
//

#ifndef SLIC_SLIC_H
#define SLIC_SLIC_H

#endif //SLIC_SLIC_H

#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

struct Pixel {
    float l = 0, a = 0, b = 0;
    float x = 0, y = 0;

    Pixel() = default;
    Pixel(float l, float a, float b, float x, float y)
        : l(l), a(a), b(b), x(x), y(y) {}
};

void rgb2lab(unsigned char R, unsigned char G, unsigned char B, float& L, float& a, float& b);
cv::Mat draw_boundaries(const cv::Mat& img, const std::vector<int>& labels);
double gradient(const cv::Mat& lab, int x, int y);
void perturbe_center(const cv::Mat& lab, int& x, int& y);
std::vector<Pixel> initialize_centers(const cv::Mat& lab, const int& K);
int get_backup_label(const std::vector<int>& new_labels, int x, int y, int width, int height);
std::vector<int> reassign_labels(const cv::Mat& lab, const std::vector<int>& labels, const int& K);

cv::Mat bgr2lab(const cv::Mat& img);
std::vector<int> find_clusters(const cv::Mat& lab, const int& K, const int& m = 10);
std::vector<int> SLIC_sequential(const cv::Mat& lab, const int& K);

cv::Mat bgr2lab_parallel(const cv::Mat& img);
std::vector<Pixel> initialize_centers_parallel(const cv::Mat& lab, const int& K);
std::vector<int> find_clusters_parallel(const cv::Mat& lab, const int& K, const int& m = 10);
std::vector<int> SLIC_parallel(const cv::Mat& lab, const int& K);