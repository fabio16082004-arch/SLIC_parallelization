#include "slic.h"
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

using clock_type = std::chrono::steady_clock;

static double ms(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::vector<double> get_time_SLIC_sequential_separated(const cv::Mat& img, const int &K) {
    std::vector<double> times;

    auto t0 = clock_type::now();
    cv::Mat lab = bgr2lab(img);
    auto t1 = clock_type::now();
    times.push_back(ms(t0, t1));

    t0 = clock_type::now();
    std::vector<int> labels = find_clusters(lab, K);
    t1 = clock_type::now();
    times.push_back(ms(t0, t1));

    t0 = clock_type::now();
    std::vector<int> new_labels = reassign_labels(lab.cols, lab.rows, labels, K);
    t1 = clock_type::now();
    times.push_back(ms(t0, t1));

    return times;
}

double get_time_SLIC_sequential(const cv::Mat& img, const int &K) {
    cv::Mat lab = bgr2lab(img);
    auto t0 = clock_type::now();
    SLIC_sequential(lab, K);
    auto t1 = clock_type::now();
    return ms(t0, t1);
}

std::vector<double> get_time_SLIC_parallel_separated(const cv::Mat& img, const int &K) {
    std::vector<double> times;

    auto t0 = clock_type::now();
    cv::Mat lab = bgr2lab_parallel(img);
    auto t1 = clock_type::now();
    times.push_back(ms(t0, t1));

    t0 = clock_type::now();
    std::vector<int> labels = find_clusters_parallel(lab, K);
    t1 = clock_type::now();
    times.push_back(ms(t0, t1));

    t0 = clock_type::now();
    std::vector<int> new_labels = reassign_labels(lab.cols, lab.rows, labels, K);
    t1 = clock_type::now();
    times.push_back(ms(t0, t1));

    return times;
}

double get_time_SLIC_parallel(const cv::Mat& img, const int &K) {
    cv::Mat lab = bgr2lab_parallel(img);
    auto t0 = clock_type::now();
    SLIC_parallel(lab, K);
    auto t1 = clock_type::now();
    return ms(t0, t1);
}

std::vector<double> get_time_SLIC_soa_separated(const cv::Mat& img, const int &K) {
    std::vector<double> times;

    auto t0 = clock_type::now();
    LabImage lab = bgr2lab_soa(img);
    auto t1 = clock_type::now();
    times.push_back(ms(t0, t1));

    t0 = clock_type::now();
    std::vector<int> labels = find_clusters_parallel_soa(lab, K);
    t1 = clock_type::now();
    times.push_back(ms(t0, t1));

    t0 = clock_type::now();
    std::vector<int> new_labels = reassign_labels(lab.width, lab.height, labels, K);
    t1 = clock_type::now();
    times.push_back(ms(t0, t1));

    return times;
}

double get_time_SLIC_soa(const cv::Mat& img, const int &K) {
    LabImage lab = bgr2lab_soa(img);
    auto t0 = clock_type::now();
    SLIC_parallel_soa(lab, K);
    auto t1 = clock_type::now();
    return ms(t0, t1);
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

int count_different(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return -1;
    int diff = 0;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] != b[i]) diff++;
    return diff;
}

int count_boundary_differences(const std::vector<int>& a, const std::vector<int>& b,
                               int width, int height) {
    int diff = 0;
    for (int y = 0; y < height - 1; y++)
        for (int x = 0; x < width - 1; x++) {
            int i = y * width + x;
            bool border_a = a[i] != a[i + 1] || a[i] != a[i + width];
            bool border_b = b[i] != b[i + 1] || b[i] != b[i + width];
            if (border_a != border_b) diff++;
        }
    return diff;
}

int count_superpixels(const std::vector<int>& labels) {
    return *std::max_element(labels.begin(), labels.end()) + 1;
}

bool diagnose(const cv::Mat& img, int K) {
    const double TOL = 0.5;

    std::cout << "===== DIAGNOSIS: AoS vs SoA, phase by phase =====\n";
    std::cout << std::fixed << std::setprecision(4);
    bool ok = true;

    cv::Mat  lab_aos = bgr2lab(img);
    LabImage lab_soa = bgr2lab_soa(img);
    int W = img.cols, H = img.rows, N = W * H;

    float max_diff = 0.0f;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            cv::Vec3f p = lab_aos.at<cv::Vec3f>(y, x);
            int idx = y * W + x;
            max_diff = std::max({ max_diff,
                                  std::abs(p[0] - lab_soa.L[idx]),
                                  std::abs(p[1] - lab_soa.A[idx]),
                                  std::abs(p[2] - lab_soa.B[idx]) });
        }
    bool ok1 = (max_diff == 0.0f);
    std::cout << "1. bgr2lab:            max difference = " << max_diff
              << (ok1 ? "   OK\n" : "   <-- ERROR\n");
    ok = ok && ok1;

    std::vector<Pixel> c_aos = initialize_centers(lab_aos, K);
    std::vector<Pixel> c_soa = initialize_centers_soa(lab_soa, K);
    bool ok2 = (c_aos.size() == c_soa.size());
    int diff_centers = 0;
    if (ok2) {
        for (size_t k = 0; k < c_aos.size(); k++)
            if (c_aos[k].x != c_soa[k].x || c_aos[k].y != c_soa[k].y)
                diff_centers++;
        ok2 = (100.0 * diff_centers / c_aos.size() < TOL);
    }
    std::cout << "2. initialize_centers: different centers = " << diff_centers
              << " of " << c_aos.size() << (ok2 ? "   OK\n" : "   <-- ERROR\n");
    ok = ok && ok2;

    std::vector<int> f_aos = find_clusters_parallel(lab_aos, K);
    std::vector<int> f_soa = find_clusters_parallel_soa(lab_soa, K);
    int d3 = count_different(f_aos, f_soa);
    bool ok3 = (d3 >= 0 && 100.0 * d3 / N < TOL);
    std::cout << "3. find_clusters:      different pixels = " << d3
              << " (" << 100.0 * d3 / N << "%)" << (ok3 ? "   OK\n" : "   <-- ERROR\n");
    ok = ok && ok3;

    std::vector<int> l_seq = SLIC_sequential(lab_aos, K);
    std::vector<int> l_aos = SLIC_parallel(lab_aos, K);
    std::vector<int> l_soa = SLIC_parallel_soa(lab_soa, K);

    int d_aos = count_different(l_seq, l_aos);
    int d_soa = count_boundary_differences(l_seq, l_soa, W, H);
    bool ok4a = (d_aos == 0);
    bool ok4b = (100.0 * d_soa / N < TOL);

    std::cout << "4. final segmentation vs sequential:\n"
              << "     AoS: " << d_aos << " different labels ("
              << 100.0 * d_aos / N << "%)" << (ok4a ? "   OK\n" : "   <-- ERROR\n")
              << "     SoA: " << d_soa << " different boundary pixels ("
              << 100.0 * d_soa / N << "%)" << (ok4b ? "   OK\n" : "   <-- ERROR\n")
              << "     superpixels: sequential " << count_superpixels(l_seq)
              << ", AoS " << count_superpixels(l_aos)
              << ", SoA " << count_superpixels(l_soa) << "\n\n";
    ok = ok && ok4a && ok4b;

    return ok;
}

int main() {
    const std::string base     = PROJECT_DIR;
    const std::string img_name = "22090";
    const std::string path     = base + "/archive/images/train/" + img_name + ".jpg";
    const int K           = 500;
    const int RUNS        = 10;
    const int MAX_THREADS = 8;

    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "Image not found in path: " << path << "\n";
        return 1;
    }

    omp_set_num_threads(MAX_THREADS);
    if (!diagnose(img, K)) {
        std::cerr << "SoA version is not correct: timings skipped.\n";
        return 1;
    }

    // ---------- 2. tempi ----------
    std::vector<double> seq = average_times(get_time_SLIC_sequential_separated,
                                            get_time_SLIC_sequential, img, K, RUNS);

    const std::vector<std::string> names = { "bgr2lab", "find_clusters", "reassign_labels", "core total" };
    const int w_name = 18, w_col = 13;
    const std::string line(w_name + 5 * w_col, '-');

    for (int n_threads = 1; n_threads <= MAX_THREADS; n_threads *= 2) {
        omp_set_num_threads(n_threads);

        std::vector<double> aos = average_times(get_time_SLIC_parallel_separated,
                                                get_time_SLIC_parallel, img, K, RUNS);
        std::vector<double> soa = average_times(get_time_SLIC_soa_separated,
                                                get_time_SLIC_soa, img, K, RUNS);

        std::cout << "===== REPORT AFTER " << RUNS << " ITERATIONS USING "
                  << n_threads << " THREADS =====\n";
        std::cout << "Image: " << img_name << " (" << img.cols << " x " << img.rows
                  << "),  K = " << K << "\n\n";

        std::cout << std::left  << std::setw(w_name) << "function"
                  << std::right << std::setw(w_col) << "sequential"
                  << std::setw(w_col) << "par AoS"
                  << std::setw(w_col) << "par SoA"
                  << std::setw(w_col) << "S AoS"
                  << std::setw(w_col) << "S SoA" << "\n";
        std::cout << line << "\n";

        std::cout << std::fixed << std::setprecision(2);
        for (size_t i = 0; i < names.size(); i++) {
            if (i == names.size() - 1) std::cout << line << "\n";
            std::cout << std::left  << std::setw(w_name) << names[i]
                      << std::right << std::setw(w_col - 3) << seq[i] << " ms"
                      << std::setw(w_col - 3) << aos[i] << " ms"
                      << std::setw(w_col - 3) << soa[i] << " ms"
                      << std::setw(w_col - 1) << seq[i] / aos[i] << "x"
                      << std::setw(w_col - 1) << seq[i] / soa[i] << "x" << "\n";
        }
        std::cout << "\n";
    }
    return 0;
}