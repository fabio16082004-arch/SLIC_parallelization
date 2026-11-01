#include "slic.h"
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <functional>
#include <chrono>
#include <algorithm>
#include <vector>
#include <string>

std::vector<double> get_time_SLIC_sequential_separated(const cv::Mat& img, const int &K) {
    std::vector<double> times;

    auto t0 = std::chrono::steady_clock::now();
    cv::Mat lab = bgr2lab(img);
    auto t1 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

    t0 = std::chrono::steady_clock::now();
    std::vector<int> labels = find_clusters(lab, K);
    t1 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

    t0 = std::chrono::steady_clock::now();
    std::vector<int> new_labels = reassign_labels(lab.cols, lab.rows, labels, K);
    t1 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return times;
}

double get_time_SLIC_sequential(const cv::Mat& img, const int &K) {
    cv::Mat lab = bgr2lab(img);

    auto t0 = std::chrono::steady_clock::now();
    SLIC_sequential(lab, K);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

std::vector<double> get_time_SLIC_parallel_separated(const cv::Mat& img, const int &K) {
    std::vector<double> times;

    auto t0 = std::chrono::steady_clock::now();
    cv::Mat lab = bgr2lab_parallel(img);
    auto t1 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

    t0 = std::chrono::steady_clock::now();
    std::vector<int> labels = find_clusters_parallel(lab, K);
    t1 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

    t0 = std::chrono::steady_clock::now();
    std::vector<int> new_labels = reassign_labels(lab.cols, lab.rows, labels, K);
    t1 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return times;
}

double get_time_SLIC_parallel(const cv::Mat& img, const int &K) {
    cv::Mat lab = bgr2lab_parallel(img);

    auto t0 = std::chrono::steady_clock::now();
    SLIC_parallel(lab, K);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

std::vector<double> average_times(
        std::vector<double> (*separated)(const cv::Mat&, const int&),
        double (*total)(const cv::Mat&, const int&),
        const cv::Mat& img, int K, int RUNS) {

    separated(img, K);
    total(img, K);

    std::vector<double> times(4, 0.0);
    for (int r = 0; r < RUNS; r++) {
        times[3] += total(img, K);
        std::vector<double> app = separated(img, K);
        for (int j = 0; j < 3; j++)
            times[j] += app[j];
    }
    for (double& t : times)
        t /= RUNS;
    return times;
}

int main() {
    setenv("OMP_DISPLAY_ENV", "TRUE", 0);
    const std::string base = PROJECT_DIR;
    const std::string img_name = "22090";
    const std::string path = base + "/archive/images/train/" + img_name + ".jpg";
    const int K = 500;
    const int RUNS = 10;
    const int MAX_THREADS = 256;

    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "Image not found in path: " << path << "\n";
        return 1;
    }

    std::vector<double> seq = average_times(get_time_SLIC_sequential_separated,
                                            get_time_SLIC_sequential, img, K, RUNS);

    const std::vector<std::string> names = { "bgr2lab", "find_clusters", "reassign_labels", "core total" };
    const int w_name = 18, w_col = 14;
    const std::string line(w_name + 3 * w_col, '-');

    for (int n_threads = 1; n_threads <= MAX_THREADS; n_threads *= 2) {
        omp_set_num_threads(n_threads);

        std::vector<double> par = average_times(get_time_SLIC_parallel_separated,
                                                get_time_SLIC_parallel, img, K, RUNS);

        std::cout << "\n===== REPORT AFTER " << RUNS << " ITERATIONS USING "
                  << n_threads << " THREADS =====\n";
        std::cout << "Image: " << img_name << " (" << img.cols << " x " << img.rows
                  << "),  K = " << K << "\n\n";

        std::cout << std::left  << std::setw(w_name) << "function"
                  << std::right << std::setw(w_col)  << "sequential"
                  << std::setw(w_col) << "parallel"
                  << std::setw(w_col) << "speed-up" << "\n";
        std::cout << line << "\n";

        std::cout << std::fixed << std::setprecision(2);
        for (size_t i = 0; i < names.size(); i++) {
            if (i == names.size() - 1) std::cout << line << "\n";
            std::cout << std::left  << std::setw(w_name) << names[i]
                      << std::right << std::setw(w_col - 3) << seq[i] << " ms"
                      << std::setw(w_col - 3) << par[i] << " ms"
                      << std::setw(w_col - 1) << seq[i] / par[i] << "x" << "\n";
        }
    }
    return 0;
}