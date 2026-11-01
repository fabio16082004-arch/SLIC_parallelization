//
// Created by fbi168 on 25/10/26.
//

#ifndef SLIC_SLIC_H
#define SLIC_SLIC_H

#include <vector>
#include <opencv2/opencv.hpp>

// ===================== strutture dati =====================

struct Pixel {
    float l = 0, a = 0, b = 0;
    float x = 0, y = 0;

    Pixel() = default;
    Pixel(float l, float a, float b, float x, float y) : l(l), a(a), b(b), x(x), y(y) {}
};

// Immagine Lab in formato SoA: un vettore per canale
struct LabImage {
    int width = 0, height = 0;
    std::vector<float> L, A, B;

    LabImage() = default;
    LabImage(int w, int h) : width(w), height(h), L(w * h), A(w * h), B(w * h) {}

    const float* L_row(int y) const { return L.data() + y * width; }
    const float* A_row(int y) const { return A.data() + y * width; }
    const float* B_row(int y) const { return B.data() + y * width; }
};

// ===================== funzioni comuni =====================

void rgb2lab(unsigned char R, unsigned char G, unsigned char B, float& L, float& a, float& b);
cv::Mat draw_boundaries(const cv::Mat& img, const std::vector<int>& labels);
int get_backup_label(const std::vector<int>& new_labels, int x, int y, int width, int height);
std::vector<int> reassign_labels(int width, int height, const std::vector<int>& labels, const int& K);

// ===================== versione sequenziale (AoS) =====================

cv::Mat bgr2lab(const cv::Mat& img);
double gradient(const cv::Mat& lab, int x, int y);
void perturbe_center(const cv::Mat& lab, int& x, int& y);
std::vector<Pixel> initialize_centers(const cv::Mat& lab, const int& K);
std::vector<int> find_clusters(const cv::Mat& lab, const int& K, const int& m = 10);
std::vector<int> SLIC_sequential(const cv::Mat& lab, const int& K);

// ===================== versione parallela (AoS, tile) =====================

cv::Mat bgr2lab_parallel(const cv::Mat& img);
std::vector<Pixel> initialize_centers_parallel(const cv::Mat& lab, const int& K);
std::vector<int> find_clusters_parallel(const cv::Mat& lab, const int& K, const int& m = 10);
std::vector<int> SLIC_parallel(const cv::Mat& lab, const int& K);

// ===================== versione parallela (SoA, tile, SIMD, accumulo locale) =====================

LabImage bgr2lab_soa(const cv::Mat& img);
double gradient_soa(const LabImage& lab, int x, int y);
void perturbe_center_soa(const LabImage& lab, int& x, int& y);
std::vector<Pixel> initialize_centers_soa(const LabImage& lab, const int& K);
std::vector<int> find_clusters_parallel_soa(const LabImage& lab, const int& K, const int& m = 10);
std::vector<int> SLIC_parallel_soa(const LabImage& lab, const int& K);

#endif // SLIC_SLIC_H