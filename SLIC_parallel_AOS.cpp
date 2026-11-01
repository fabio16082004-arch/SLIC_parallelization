//
// Created by fbi168 on 25/10/26.
//
#include <iostream>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <limits>
#include <algorithm>
#include <omp.h>
#include "slic.h"


cv::Mat bgr2lab_parallel(const cv::Mat& img) {
    cv::Mat lab(img.rows, img.cols, CV_32FC3);

    #pragma omp parallel for default(none) shared(img, lab)
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

std::vector<Pixel> initialize_centers_parallel(const cv::Mat& lab, const int &K) {
    int width = lab.cols, height = lab.rows;
    int N = width * height;
    int S = static_cast<int>(std::lround(std::sqrt(static_cast<double>(N) / K)));

    int cols = static_cast<int>(std::lround(static_cast<double>(width)  / S));
    int rows = static_cast<int>(std::lround(static_cast<double>(height) / S));

    double stepX = static_cast<double>(width)  / cols;
    double stepY = static_cast<double>(height) / rows;

    std::vector<Pixel> centers(rows * cols);

    #pragma omp parallel for default(none) shared(lab, centers, rows, cols, stepX, stepY)
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            int x = static_cast<int>(stepX * (0.5 + i));
            int y = static_cast<int>(stepY * (0.5 + j));

            perturbe_center(lab, x, y);

            cv::Vec3f p = lab.at<cv::Vec3f>(y, x);

            centers[j * cols + i] = Pixel(p[0], p[1], p[2], x, y);
        }
    }
    return centers;
}

std::vector<int> find_clusters_parallel_1(const cv::Mat& lab, const int &K, const int &m) {
    std::vector<Pixel> centers = initialize_centers(lab, K);

    int width = lab.cols, height = lab.rows;
    int N = width * height;
    int S = static_cast<int>(std::lround(std::sqrt(static_cast<double>(N) / K)));
    int Kc = static_cast<int>(centers.size());

    std::vector<int>    label(N, -1);
    std::vector<double> dist(N);

    std::vector<double> sum_l_vec(Kc), sum_a_vec(Kc), sum_b_vec(Kc);
    std::vector<double> sum_x_vec(Kc), sum_y_vec(Kc);
    std::vector<int>    count_vec(Kc);

    double *sum_l = sum_l_vec.data(), *sum_a = sum_a_vec.data(), *sum_b = sum_b_vec.data();
    double *sum_x = sum_x_vec.data(), *sum_y = sum_y_vec.data();
    int    *count = count_vec.data();

    #pragma omp parallel default(none) \
            shared(lab, centers, label, dist, sum_l, sum_a, sum_b, sum_x, sum_y, count, \
                   width, height, N, S, Kc, m)
    for (int i = 0; i < 10; i++) {

        #pragma omp for
        for (int idx = 0; idx < N; idx++)
            dist[idx] = std::numeric_limits<double>::max();

        for (int k = 0; k < Kc; k++) {
            const Pixel c = centers[k];
            int cx = static_cast<int>(c.x), cy = static_cast<int>(c.y);

            int x0 = std::max(0, cx - S), x1 = std::min(width  - 1, cx + S);
            int y0 = std::max(0, cy - S), y1 = std::min(height - 1, cy + S);

            #pragma omp for
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

        #pragma omp for
        for (int idx = 0; idx < Kc; idx++) {
            sum_l[idx] = sum_a[idx] = sum_b[idx] = sum_x[idx] = sum_y[idx] = 0.0;
            count[idx] = 0;
        }

        #pragma omp for reduction(+: sum_l[0:Kc], sum_a[0:Kc], sum_b[0:Kc], \
                                     sum_x[0:Kc], sum_y[0:Kc], count[0:Kc])
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

        #pragma omp for
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

std::vector<int> find_clusters_parallel_2(const cv::Mat& lab, const int &K, const int &m) {
    std::vector<Pixel> centers = initialize_centers(lab, K);

    int width = lab.cols, height = lab.rows;
    int N = width * height;
    int S = static_cast<int>(std::lround(std::sqrt(static_cast<double>(N) / K)));
    int Kc = static_cast<int>(centers.size());

    int cols = static_cast<int>(std::lround(static_cast<double>(width)  / S));
    int rows = static_cast<int>(std::lround(static_cast<double>(height) / S));
    int n_cells = cols * rows;

    double stepX = static_cast<double>(width)  / cols;
    double stepY = static_cast<double>(height) / rows;
    double w = static_cast<double>(m) / S;

    std::vector<int> label(N, -1);

    std::vector<int> centers_occurrencies(n_cells, 0);
    std::vector<int> cell_start(n_cells + 1, 0);
    std::vector<int> cursor(n_cells + 1, 0);
    std::vector<int> cell_centers(Kc);

    std::vector<double> sum_l_vec(Kc), sum_a_vec(Kc), sum_b_vec(Kc);
    std::vector<double> sum_x_vec(Kc), sum_y_vec(Kc);
    std::vector<int>    count_vec(Kc);

    double *sum_l = sum_l_vec.data(), *sum_a = sum_a_vec.data(), *sum_b = sum_b_vec.data();
    double *sum_x = sum_x_vec.data(), *sum_y = sum_y_vec.data();
    int    *count = count_vec.data();

    #pragma omp parallel default(none) \
            shared(lab, centers, label, sum_l, sum_a, sum_b, sum_x, sum_y, count, \
                   cell_start, cell_centers, centers_occurrencies, cursor) \
            firstprivate(width, height, S, Kc, rows, cols, n_cells, stepX, stepY, w)
    for (int i = 0; i < 10; i++) {

        #pragma omp single
        {
            std::fill(centers_occurrencies.begin(), centers_occurrencies.end(), 0);

            for (int idx = 0; idx < Kc; idx++) {
                sum_l[idx] = sum_a[idx] = sum_b[idx] = sum_x[idx] = sum_y[idx] = 0.0;
                count[idx] = 0;
            }

            for (int k = 0; k < Kc; k++) {
                const Pixel& c = centers[k];
                int ci = static_cast<int>(c.x / stepX);
                int cj = static_cast<int>(c.y / stepY);
                centers_occurrencies[cj * cols + ci]++;
            }

            cell_start[0] = 0;
            for (int cell = 1; cell <= n_cells; cell++)
                cell_start[cell] = cell_start[cell - 1] + centers_occurrencies[cell - 1];

            cursor = cell_start;
            for (int k = 0; k < Kc; k++) {
                const Pixel& c = centers[k];
                int ci = static_cast<int>(c.x / stepX);
                int cj = static_cast<int>(c.y / stepY);
                cell_centers[cursor[cj * cols + ci]++] = k;
            }
        }

        #pragma omp for reduction(+: sum_l[0:Kc], sum_a[0:Kc], sum_b[0:Kc], \
                                     sum_x[0:Kc], sum_y[0:Kc], count[0:Kc])
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                cv::Vec3f p = lab.at<cv::Vec3f>(y, x);

                int gi = static_cast<int>(x / stepX);
                int gj = static_cast<int>(y / stepY);

                double best = std::numeric_limits<double>::max();
                int best_k = -1;

                for (int nj = std::max(0, gj - 2); nj <= std::min(gj + 2, rows - 1); nj++) {
                    for (int ni = std::max(0, gi - 2); ni <= std::min(gi + 2, cols - 1); ni++) {
                        int cell = nj * cols + ni;

                        for (int t = cell_start[cell]; t < cell_start[cell + 1]; t++) {
                            int k = cell_centers[t];
                            const Pixel& c = centers[k];

                            int cx = static_cast<int>(c.x), cy = static_cast<int>(c.y);
                            if (x < cx - S || x > cx + S || y < cy - S || y > cy + S) continue;

                            double dl = p[0] - c.l, da = p[1] - c.a, db = p[2] - c.b;
                            double d_lab = std::sqrt(dl*dl + da*da + db*db);

                            double dx = x - c.x, dy = y - c.y;
                            double d_xy = std::sqrt(dx*dx + dy*dy);

                            double D = d_lab + w * d_xy;
                            if (D < best || (D == best && k < best_k)) {
                                best = D;
                                best_k = k;
                            }
                        }
                    }
                }

                label[y * width + x] = best_k;
                if (best_k >= 0) {
                    sum_l[best_k] += p[0];
                    sum_a[best_k] += p[1];
                    sum_b[best_k] += p[2];
                    sum_x[best_k] += x;
                    sum_y[best_k] += y;
                    count[best_k]++;
                }
            }
        }

        #pragma omp for
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

std::vector<int> find_clusters_parallel(const cv::Mat& lab, const int &K, const int &m) {
    std::vector<Pixel> centers = initialize_centers(lab, K);

    int width = lab.cols, height = lab.rows;
    int N = width * height;
    int S = static_cast<int>(std::lround(std::sqrt(static_cast<double>(N) / K)));
    int Kc = static_cast<int>(centers.size());

    int cols = static_cast<int>(std::lround(static_cast<double>(width)  / S));
    int rows = static_cast<int>(std::lround(static_cast<double>(height) / S));
    int n_cells = cols * rows;

    double stepX = static_cast<double>(width)  / cols;
    double stepY = static_cast<double>(height) / rows;
    double w = static_cast<double>(m) / S;

    std::vector<int> label(N, -1);

    std::vector<int> centers_occurrencies(n_cells, 0);
    std::vector<int> cell_start(n_cells + 1, 0);
    std::vector<int> cursor(n_cells + 1, 0);
    std::vector<int> cell_centers(Kc);

    std::vector<double> sum_l_vec(Kc), sum_a_vec(Kc), sum_b_vec(Kc);
    std::vector<double> sum_x_vec(Kc), sum_y_vec(Kc);
    std::vector<int>    count_vec(Kc);

    double *sum_l = sum_l_vec.data(), *sum_a = sum_a_vec.data(), *sum_b = sum_b_vec.data();
    double *sum_x = sum_x_vec.data(), *sum_y = sum_y_vec.data();
    int    *count = count_vec.data();

       #pragma omp parallel default(none) \
            shared(lab, centers, label, sum_l, sum_a, sum_b, sum_x, sum_y, count, \
                   cell_start, cell_centers, centers_occurrencies, cursor) \
            firstprivate(width, height, S, Kc, rows, cols, n_cells, stepX, stepY, w)
    {
        std::vector<int>    candidate_centers;
        std::vector<double> tile_dist;

        for (int i = 0; i < 10; i++) {

            #pragma omp single
            {
                std::fill(centers_occurrencies.begin(), centers_occurrencies.end(), 0);

                for (int idx = 0; idx < Kc; idx++) {
                    sum_l[idx] = sum_a[idx] = sum_b[idx] = sum_x[idx] = sum_y[idx] = 0.0;
                    count[idx] = 0;
                }

                for (int k = 0; k < Kc; k++) {
                    const Pixel& c = centers[k];
                    int ci = static_cast<int>(c.x / stepX);
                    int cj = static_cast<int>(c.y / stepY);
                    centers_occurrencies[cj * cols + ci]++;
                }

                cell_start[0] = 0;
                for (int cell = 1; cell <= n_cells; cell++)
                    cell_start[cell] = cell_start[cell - 1] + centers_occurrencies[cell - 1];

                cursor = cell_start;
                for (int k = 0; k < Kc; k++) {
                    const Pixel& c = centers[k];
                    int ci = static_cast<int>(c.x / stepX);
                    int cj = static_cast<int>(c.y / stepY);
                    cell_centers[cursor[cj * cols + ci]++] = k;
                }
            }

            #pragma omp for collapse(2) schedule(dynamic) \
                    reduction(+: sum_l[0:Kc], sum_a[0:Kc], sum_b[0:Kc], \
                                 sum_x[0:Kc], sum_y[0:Kc], count[0:Kc])
            for (int gj = 0; gj < rows; gj++) {
                for (int gi = 0; gi < cols; gi++) {

                    int x_begin = static_cast<int>(std::ceil(gi * stepX));
                    int x_end   = (gi == cols - 1) ? width  : static_cast<int>(std::ceil((gi + 1) * stepX));
                    int y_begin = static_cast<int>(std::ceil(gj * stepY));
                    int y_end   = (gj == rows - 1) ? height : static_cast<int>(std::ceil((gj + 1) * stepY));
                    int tw = x_end - x_begin, th = y_end - y_begin;

                    candidate_centers.clear();
                    for (int nj = std::max(0, gj - 2); nj <= std::min(gj + 2, rows - 1); nj++) {
                        for (int ni = std::max(0, gi - 2); ni <= std::min(gi + 2, cols - 1); ni++) {
                            int cell = nj * cols + ni;
                            for (int t = cell_start[cell]; t < cell_start[cell + 1]; t++)
                                candidate_centers.push_back(cell_centers[t]);
                        }
                    }
                    std::sort(candidate_centers.begin(), candidate_centers.end());

                    tile_dist.assign(static_cast<size_t>(tw) * th, std::numeric_limits<double>::max());
                    for (int y0 = y_begin; y0 < y_end; y0++)
                        for (int x0 = x_begin; x0 < x_end; x0++)
                            label[y0 * width + x0] = -1;

                    for (int k : candidate_centers) {
                        const Pixel& c = centers[k];
                        int cx = static_cast<int>(c.x), cy = static_cast<int>(c.y);

                        int xa = std::max(x_begin, cx - S);
                        int xb = std::min(x_end - 1, cx + S);
                        int ya = std::max(y_begin, cy - S);
                        int yb = std::min(y_end - 1, cy + S);
                        if (xa > xb || ya > yb) continue;

                        for (int y0 = ya; y0 <= yb; y0++) {
                            for (int x0 = xa; x0 <= xb; x0++) {
                                cv::Vec3f p = lab.at<cv::Vec3f>(y0, x0);

                                double dl = p[0] - c.l, da = p[1] - c.a, db = p[2] - c.b;
                                double d_lab = std::sqrt(dl*dl + da*da + db*db);

                                double dx = x0 - c.x, dy = y0 - c.y;
                                double d_xy = std::sqrt(dx*dx + dy*dy);

                                double D = d_lab + w * d_xy;

                                int tidx = (y0 - y_begin) * tw + (x0 - x_begin);
                                if (D < tile_dist[tidx]) {
                                    tile_dist[tidx]        = D;
                                    label[y0 * width + x0] = k;
                                }
                            }
                        }
                    }

                    for (int y0 = y_begin; y0 < y_end; y0++) {
                        for (int x0 = x_begin; x0 < x_end; x0++) {
                            int k = label[y0 * width + x0];
                            if (k < 0) continue;

                            cv::Vec3f p = lab.at<cv::Vec3f>(y0, x0);
                            sum_l[k] += p[0];
                            sum_a[k] += p[1];
                            sum_b[k] += p[2];
                            sum_x[k] += x0;
                            sum_y[k] += y0;
                            count[k]++;
                        }
                    }
                }
            }

            #pragma omp for
            for (int idx = 0; idx < Kc; idx++) {
                if (count[idx] == 0) continue;
                centers[idx].l = sum_l[idx] / count[idx];
                centers[idx].a = sum_a[idx] / count[idx];
                centers[idx].b = sum_b[idx] / count[idx];
                centers[idx].x = sum_x[idx] / count[idx];
                centers[idx].y = sum_y[idx] / count[idx];
            }
        }
    }

    return label;
}


int get_backup_label(const std::vector<int>& new_labels,
                     int x, int y, int width, int height) {
    if (x > 0 && new_labels[y * width + (x - 1)] != -1) return new_labels[y * width + (x - 1)];
    if (y > 0 && new_labels[(y - 1) * width + x] != -1) return new_labels[(y - 1) * width + x];
    if (x < width - 1  && new_labels[y * width + (x + 1)] != -1) return new_labels[y * width + (x + 1)];
    if (y < height - 1 && new_labels[(y + 1) * width + x] != -1) return new_labels[(y + 1) * width + x];
    return -1;
}

std::vector<int> reassign_labels(int width, int height, const std::vector<int>& labels, const int &K) {
    int N = width * height;
    double threshold = static_cast<double>(N) / (4.0 * K);
    std::vector<int> segment;

    std::vector<int> new_labels(N, -1);
    int label = 0;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {

            int start_idx = j * width + i;
            if (new_labels[start_idx] != -1) continue;

            int old_label = labels[start_idx];
            int backup    = get_backup_label(new_labels, i, j, width, height);

            segment.clear();

            new_labels[start_idx] = label;
            segment.push_back(start_idx);
            size_t head = 0;

            while (head < segment.size()) {
                int idx = segment[head];
                head++;

                int y = idx / width;
                int x = idx % width;
                if (x > 0) {
                    int n = y * width + (x - 1);
                    if (new_labels[n] == -1 && labels[n] == old_label) {
                        new_labels[n] = label;
                        segment.push_back(n);
                    }
                }
                if (y > 0) {
                    int n = (y - 1) * width + x;
                    if (new_labels[n] == -1 && labels[n] == old_label) {
                        new_labels[n] = label;
                        segment.push_back(n);
                    }
                }
                if (x < width - 1) {
                    int n = y * width + (x + 1);
                    if (new_labels[n] == -1 && labels[n] == old_label) {
                        new_labels[n] = label;
                        segment.push_back(n);
                    }

                }
                if (y < height - 1) {
                    int n = (y + 1) * width + x;
                    if (new_labels[n] == -1 && labels[n] == old_label) {
                        new_labels[n] = label;
                        segment.push_back(n);
                    }
                }
            }

            if (static_cast<double>(segment.size()) < threshold && backup != -1) {
                for (int n : segment) new_labels[n] = backup;
            } else {
                label++;
            }
        }
    }

    return new_labels;
}

std::vector<int> SLIC_parallel(const cv::Mat& lab, const int &K) {
    std::vector<int> labels = find_clusters_parallel(lab, K);
    std::vector<int> new_labels = reassign_labels(lab.cols, lab.rows, labels, K);
    return new_labels;
}