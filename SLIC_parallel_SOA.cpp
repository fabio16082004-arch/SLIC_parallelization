//
// Created by fbi168 on 01/11/26.
//
// Versione parallela di SLIC con immagine in formato SoA e ciclo interno vettorizzato.
//

#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "slic.h"

LabImage bgr2lab_soa(const cv::Mat& img) {
    LabImage lab(img.cols, img.rows);

    #pragma omp parallel for default(none) shared(img, lab)
    for (int y = 0; y < img.rows; y++) {
        const cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < img.cols; x++) {
            int idx = y * img.cols + x;
            rgb2lab(row[x][2], row[x][1], row[x][0], lab.L[idx], lab.A[idx], lab.B[idx]);
        }
    }
    return lab;
}

double gradient_soa(const LabImage& lab, int x, int y) {
    int W = lab.width;
    int r = y * W + (x + 1), l = y * W + (x - 1);       // destra, sinistra
    int d = (y + 1) * W + x, u = (y - 1) * W + x;       // sotto, sopra

    float dxL = lab.L[r] - lab.L[l], dxA = lab.A[r] - lab.A[l], dxB = lab.B[r] - lab.B[l];
    float dyL = lab.L[d] - lab.L[u], dyA = lab.A[d] - lab.A[u], dyB = lab.B[d] - lab.B[u];

    float gx = dxL * dxL + dxA * dxA + dxB * dxB;
    float gy = dyL * dyL + dyA * dyA + dyB * dyB;
    return gx + gy;
}

void perturbe_center_soa(const LabImage& lab, int &x, int &y) {
    int min_x = x, min_y = y;
    double grad = gradient_soa(lab, x, y);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int nx = x + (i - 1);
            int ny = y + (j - 1);
            if (nx < 1 || nx > lab.width - 2 || ny < 1 || ny > lab.height - 2) continue;

            double g = gradient_soa(lab, nx, ny);
            if (g < grad) {
                min_x = nx;
                min_y = ny;
                grad = g;
            }
        }
    }
    x = min_x;
    y = min_y;
}

std::vector<Pixel> initialize_centers_soa(const LabImage& lab, const int &K) {
    int width = lab.width, height = lab.height;
    int N = width * height;
    int S = static_cast<int>(std::lround(std::sqrt(static_cast<double>(N) / K)));

    int cols = static_cast<int>(std::lround(static_cast<double>(width)  / S));
    int rows = static_cast<int>(std::lround(static_cast<double>(height) / S));

    double stepX = static_cast<double>(width)  / cols;
    double stepY = static_cast<double>(height) / rows;

    std::vector<Pixel> centers(rows * cols);

    #pragma omp parallel for default(none) shared(lab, centers, rows, cols, stepX, stepY, width)
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            int x = static_cast<int>(stepX * (0.5 + i));
            int y = static_cast<int>(stepY * (0.5 + j));

            perturbe_center_soa(lab, x, y);

            int idx = y * width + x;
            centers[j * cols + i] = Pixel(lab.L[idx], lab.A[idx], lab.B[idx], x, y);
        }
    }
    return centers;
}

std::vector<int> find_clusters_parallel_soa(const LabImage& lab, const int &K, const int &m) {
    std::vector<Pixel> centers = initialize_centers_soa(lab, K);

    int width = lab.width, height = lab.height;
    int N = width * height;
    int S = static_cast<int>(std::lround(std::sqrt(static_cast<double>(N) / K)));
    int Kc = static_cast<int>(centers.size());

    int cols = static_cast<int>(std::lround(static_cast<double>(width)  / S));
    int rows = static_cast<int>(std::lround(static_cast<double>(height) / S));
    int n_cells = cols * rows;

    double stepX = static_cast<double>(width)  / cols;
    double stepY = static_cast<double>(height) / rows;
    float  w = static_cast<float>(m) / S;

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
    int    *lab_out = label.data();

    #pragma omp parallel default(none) \
            shared(lab, centers, lab_out, sum_l, sum_a, sum_b, sum_x, sum_y, count, \
                   cell_start, cell_centers, centers_occurrencies, cursor) \
            firstprivate(width, height, S, Kc, rows, cols, n_cells, stepX, stepY, w)
    {
        std::vector<int>    candidate_centers;
        std::vector<float>  tile_dist;
        std::vector<int>    tile_lab;
        std::vector<double> ls_l, ls_a, ls_b, ls_x, ls_y;
        std::vector<int>    ls_n;

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
                    for (int nj = std::max(0, gj - 2); nj <= std::min(gj + 2, rows - 1); nj++)
                        for (int ni = std::max(0, gi - 2); ni <= std::min(gi + 2, cols - 1); ni++) {
                            int cell = nj * cols + ni;
                            for (int t = cell_start[cell]; t < cell_start[cell + 1]; t++)
                                candidate_centers.push_back(cell_centers[t]);
                        }
                    std::sort(candidate_centers.begin(), candidate_centers.end());
                    int nc = static_cast<int>(candidate_centers.size());

                    tile_dist.assign(static_cast<size_t>(tw) * th, std::numeric_limits<float>::max());
                    tile_lab.assign(static_cast<size_t>(tw) * th, -1);

                    for (int ci = 0; ci < nc; ci++) {
                        const Pixel& c = centers[candidate_centers[ci]];
                        int cx = static_cast<int>(c.x), cy = static_cast<int>(c.y);

                        int xa = std::max(x_begin, cx - S);
                        int xb = std::min(x_end - 1, cx + S);
                        int ya = std::max(y_begin, cy - S);
                        int yb = std::min(y_end - 1, cy + S);
                        if (xa > xb || ya > yb) continue;

                        const float cl = c.l, ca = c.a, cb = c.b, ccx = c.x, ccy = c.y;

                        for (int y0 = ya; y0 <= yb; y0++) {
                            const float* Lr = lab.L_row(y0);
                            const float* Ar = lab.A_row(y0);
                            const float* Br = lab.B_row(y0);
                            float* dr = tile_dist.data() + (y0 - y_begin) * tw - x_begin;
                            int*   tl = tile_lab.data()  + (y0 - y_begin) * tw - x_begin;
                            const float dy = static_cast<float>(y0) - ccy;

                            #pragma omp simd
                            for (int x0 = xa; x0 <= xb; x0++) {
                                float dl = Lr[x0] - cl, da = Ar[x0] - ca, db = Br[x0] - cb;
                                float dx = static_cast<float>(x0) - ccx;
                                float D  = std::sqrt(dl*dl + da*da + db*db)
                                         + w * std::sqrt(dx*dx + dy*dy);

                                bool better = D < dr[x0];
                                dr[x0] = better ? D  : dr[x0];
                                tl[x0] = better ? ci : tl[x0];
                            }
                        }
                    }

                    ls_l.assign(nc, 0.0); ls_a.assign(nc, 0.0); ls_b.assign(nc, 0.0);
                    ls_x.assign(nc, 0.0); ls_y.assign(nc, 0.0); ls_n.assign(nc, 0);

                    for (int y0 = y_begin; y0 < y_end; y0++) {
                        const float* Lr = lab.L_row(y0);
                        const float* Ar = lab.A_row(y0);
                        const float* Br = lab.B_row(y0);
                        const int*   tl = tile_lab.data() + (y0 - y_begin) * tw - x_begin;
                        int*         lr = lab_out + y0 * width;

                        for (int x0 = x_begin; x0 < x_end; x0++) {
                            int ci = tl[x0];
                            lr[x0] = (ci >= 0) ? candidate_centers[ci] : -1;
                            if (ci < 0) continue;
                            ls_l[ci] += Lr[x0];
                            ls_a[ci] += Ar[x0];
                            ls_b[ci] += Br[x0];
                            ls_x[ci] += x0;
                            ls_y[ci] += y0;
                            ls_n[ci]++;
                        }
                    }

                    for (int ci = 0; ci < nc; ci++) {
                        if (ls_n[ci] == 0) continue;
                        int k = candidate_centers[ci];
                        sum_l[k] += ls_l[ci];
                        sum_a[k] += ls_a[ci];
                        sum_b[k] += ls_b[ci];
                        sum_x[k] += ls_x[ci];
                        sum_y[k] += ls_y[ci];
                        count[k] += ls_n[ci];
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

std::vector<int> SLIC_parallel_soa(const LabImage& lab, const int &K) {
    std::vector<int> labels = find_clusters_parallel_soa(lab, K);
    return reassign_labels(lab.width, lab.height, labels, K);
}