#include <iostream>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <limits>
#include <algorithm>
#include "slic.h"

cv::Mat draw_boundaries(const cv::Mat& img, const std::vector<int>& labels) {
    cv::Mat out = img.clone();
    int W = img.cols, H = img.rows;
    for (int y = 0; y < H - 1; y++) {
        for (int x = 0; x < W - 1; x++) {
            int l = labels[y * W + x];
            if (l != labels[y * W + x + 1] || l != labels[(y + 1) * W + x])
                out.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
        }
    }
    return out;
}


float srgb_to_linear(unsigned char v) {
    float c = v / 255.0f;
    return c <= 0.04045f ? c / 12.92f
                         : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

float lab_f(float t) {
    return t > 0.008856f ? std::cbrt(t)
                         : 7.787f * t + 16.0f / 116.0f;
}

void rgb2lab(unsigned char R, unsigned char G, unsigned char B,
             float& L, float& a, float& b) {
    float r  = srgb_to_linear(R);
    float g  = srgb_to_linear(G);
    float bl = srgb_to_linear(B);

    float X = (0.4124564f * r + 0.3575761f * g + 0.1804375f * bl) / 0.95047f;
    float Y = (0.2126729f * r + 0.7151522f * g + 0.0721750f * bl) / 1.00000f;
    float Z = (0.0193339f * r + 0.1191920f * g + 0.9503041f * bl) / 1.08883f;

    // 3. XYZ -> Lab
    float fx = lab_f(X);
    float fy = lab_f(Y);
    float fz = lab_f(Z);

    L = 116.0f * fy - 16.0f;
    a = 500.0f * (fx - fy);
    b = 200.0f * (fy - fz);
}
cv::Mat bgr2lab(const cv::Mat& img) {
    cv::Mat lab(img.rows, img.cols, CV_32FC3);

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            cv::Vec3b p = img.at<cv::Vec3b>(y, x);
            float L, a, b;
            rgb2lab(p[2], p[1], p[0], L, a, b);
            lab.at<cv::Vec3f>(y, x) = cv::Vec3f(L, a, b);
        }
    }
    return lab;
}

/*STEP - 1 & 2: INITIALISING CENTERS*/
double gradient(const cv::Mat& lab, int x, int y) {
    cv::Vec3f dx = lab.at<cv::Vec3f>(y, x + 1) - lab.at<cv::Vec3f>(y, x - 1);
    cv::Vec3f dy = lab.at<cv::Vec3f>(y + 1, x) - lab.at<cv::Vec3f>(y - 1, x);
    return dx.dot(dx) + dy.dot(dy);
}

void perturbe_center(const cv::Mat& lab, int &x, int &y) {
    int min_x = x;
    int min_y = y;
    double grad = gradient(lab, x, y);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int nx = x + (i - 1);
            int ny = y + (j - 1);
            if (nx < 1 || nx > lab.cols - 2 || ny < 1 || ny > lab.rows - 2) continue;
            if (grad > gradient(lab, nx, ny)) {
                min_x = nx;
                min_y = ny;
                grad = gradient(lab, nx, ny);
            }
        }
    }
    x = min_x;
    y = min_y;
}

std::vector<Pixel> initialize_centers(const cv::Mat& lab, const int &K) {
    int width = lab.cols, height = lab.rows;
    int N = width * height;
    int S = static_cast<int>(std::lround(std::sqrt(static_cast<double>(N) / K)));

    int cols = static_cast<int>(std::lround(static_cast<double>(width)  / S));
    int rows = static_cast<int>(std::lround(static_cast<double>(height) / S));

    double stepX = static_cast<double>(width)  / cols;
    double stepY = static_cast<double>(height) / rows;

    std::vector<Pixel> centers;
    centers.reserve(rows * cols);

    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            int x = static_cast<int>(stepX * (0.5 + i));
            int y = static_cast<int>(stepY * (0.5 + j));

            perturbe_center(lab, x, y);

            cv::Vec3f p = lab.at<cv::Vec3f>(y, x);

            centers.emplace_back(p[0], p[1], p[2], x, y);
        }
    }
    return centers;
}


std::vector<int> find_clusters(const cv::Mat& lab, const int &K, const int &m) {
    std::vector<Pixel> centers = initialize_centers(lab, K);

    int width = lab.cols, height = lab.rows;
    int N = width * height;
    int S = static_cast<int>(std::lround(std::sqrt(static_cast<double>(N) / K)));

    std::vector<int> label(N, -1);
    for (int i = 0; i < 10; i++) {
        std::vector<double> dist(N, std::numeric_limits<double>::max());
        for (int k = 0; k < static_cast<int>(centers.size()); k++) {
            const Pixel& c = centers[k];
            int cx = static_cast<int>(c.x), cy = static_cast<int>(c.y);

            int x0 = std::max(0, cx - S), x1 = std::min(width  - 1, cx + S);
            int y0 = std::max(0, cy - S), y1 = std::min(height - 1, cy + S);

            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) {
                    cv::Vec3f p = lab.at<cv::Vec3f>(y, x);

                    double dl = p[0] - c.l, da = p[1] - c.a, db = p[2] - c.b;
                    double d_lab = std::sqrt(dl*dl + da*da + db*db);

                    double dx = x - c.x, dy = y - c.y;
                    double d_xy = std::sqrt(dx*dx + dy*dy);

                    double D = d_lab + static_cast<double>(m) / S * d_xy;

                    int idx = y * width + x;
                    if (D < dist[idx]) {
                        dist[idx]  = D;
                        label[idx] = k;
                    }
                }
            }
        }

        int Kc = static_cast<int>(centers.size());

        std::vector<double> sum_l(Kc, 0), sum_a(Kc, 0), sum_b(Kc, 0);
        std::vector<double> sum_x(Kc, 0), sum_y(Kc, 0);
        std::vector<int> count(Kc, 0);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                cv::Vec3f p = lab.at<cv::Vec3f>(y, x);
                int idx = label[y * width + x];
                if (idx < 0) continue;
                sum_l[idx] += p[0];
                sum_a[idx] += p[1];
                sum_b[idx] += p[2];
                sum_x[idx] += x;
                sum_y[idx] += y;
                count[idx]++;
            }
        }

        for (int idx = 0; idx < Kc; idx++) {
            if (count[idx] == 0) continue;
            centers[idx].l = sum_l[idx] / count[idx];
            centers[idx].a = sum_a[idx] / count[idx];
            centers[idx].b = sum_b[idx] / count[idx];
            centers[idx].x = sum_x[idx] / count[idx];
            centers[idx].y = sum_y[idx] / count[idx];
        }

    }
    return label;
}

std::vector<int> SLIC_sequential(const cv::Mat& lab, const int &K) {
    std::vector<int> labels = find_clusters(lab, K);
    std::vector<int> new_labels = reassign_labels(lab.cols, lab.rows, labels, K);
    return new_labels;
}