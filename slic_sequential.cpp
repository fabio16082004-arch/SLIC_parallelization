#include <iostream>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <limits>
#include <algorithm>

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


void rgb2lab(unsigned char R, unsigned char G, unsigned char B,
             float& L, float& a, float& b) {
    auto lin = [](float c) {
        c /= 255.0f;
        return c <= 0.04045f ? c / 12.92f
                             : std::pow((c + 0.055f) / 1.055f, 2.4f);
    };
    float r = lin(R), g = lin(G), bl = lin(B);

    float X = (0.4124564f*r + 0.3575761f*g + 0.1804375f*bl) / 0.95047f;
    float Y = (0.2126729f*r + 0.7151522f*g + 0.0721750f*bl) / 1.00000f;
    float Z = (0.0193339f*r + 0.1191920f*g + 0.9503041f*bl) / 1.08883f;

    auto f = [](float t) {
        return t > 0.008856f ? std::cbrt(t) : 7.787f*t + 16.0f/116.0f;
    };
    float fx = f(X), fy = f(Y), fz = f(Z);

    L = 116.0f*fy - 16.0f;
    a = 500.0f*(fx - fy);
    b = 200.0f*(fy - fz);
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

struct Pixel {
    float l = 0, a = 0, b = 0;
    float x = 0, y = 0;

    Pixel() = default;
    Pixel(float l, float a, float b, float x, float y)
        : l(l), a(a), b(b), x(x), y(y) {}
};

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

std::vector<Pixel> inizialize_centers(const cv::Mat& lab, const int &K) {   // lab = immagine Lab in float
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


std::vector<int> find_clusters(const cv::Mat& lab, const int &K, const int &m = 10) {
    std::vector<Pixel> centers = inizialize_centers(lab, K);

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

int get_backup_label(const std::vector<int>& new_labels,
                     int x, int y, int width, int height) {
    if (x > 0 && new_labels[y * width + (x - 1)] != -1) return new_labels[y * width + (x - 1)];
    if (y > 0 && new_labels[(y - 1) * width + x] != -1) return new_labels[(y - 1) * width + x];
    if (x < width - 1  && new_labels[y * width + (x + 1)] != -1) return new_labels[y * width + (x + 1)];
    if (y < height - 1 && new_labels[(y + 1) * width + x] != -1) return new_labels[(y + 1) * width + x];
    return -1;
}

std::vector<int> reassign_labels(const cv::Mat& lab, const std::vector<int>& labels, const int &K) {
    int width = lab.cols, height = lab.rows;
    int N = width * height;
    double threshold = static_cast<double>(N) / (4.0 * K);

    std::vector<int> new_labels(N, -1);
    int label = 0;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {

            int start_idx = j * width + i;
            if (new_labels[start_idx] != -1) continue;

            int old_label = labels[start_idx];
            int backup    = get_backup_label(new_labels, i, j, width, height);

            std::vector<int> segment;
            std::queue<std::pair<int, int>> q;

            new_labels[start_idx] = label;
            segment.push_back(start_idx);
            q.emplace(j, i);

            while (!q.empty()) {
                auto [y, x] = q.front();
                q.pop();

                if (x > 0) {
                    int n = y * width + (x - 1);
                    if (new_labels[n] == -1 && labels[n] == old_label) {
                        new_labels[n] = label;
                        segment.push_back(n);
                        q.emplace(y, x - 1);
                    }
                }
                if (y > 0) {
                    int n = (y - 1) * width + x;
                    if (new_labels[n] == -1 && labels[n] == old_label) {
                        new_labels[n] = label;
                        segment.push_back(n);
                        q.emplace(y - 1, x);
                    }
                }
                if (x < width - 1) {
                    int n = y * width + (x + 1);
                    if (new_labels[n] == -1 && labels[n] == old_label) {
                        new_labels[n] = label;
                        segment.push_back(n);
                        q.emplace(y, x + 1);
                    }
                }
                if (y < height - 1) {
                    int n = (y + 1) * width + x;
                    if (new_labels[n] == -1 && labels[n] == old_label) {
                        new_labels[n] = label;
                        segment.push_back(n);
                        q.emplace(y + 1, x);
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

std::vector<int> SLIC_sequential(const cv::Mat& lab, const int &K) {
    std::vector<int> labels = find_clusters(lab, K);
    std::vector<int> new_labels = reassign_labels(lab, labels, K);
    return new_labels;
}

int main() {
    const std::string base = PROJECT_DIR;
    const std::string path = base + "/archive/images/train/22090.jpg";
    const int K = 250;

    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "Immagine non trovata: " << path << "\n";
        return 1;
    }
    std::cout << "Immagine: " << img.cols << " x " << img.rows
              << " (N = " << img.cols * img.rows << ")\n";

    cv::Mat lab = bgr2lab(img);

    cv::Vec3f p = lab.at<cv::Vec3f>(0, 0);
    std::cout << "Pixel (0,0) in Lab: L=" << p[0] << " a=" << p[1] << " b=" << p[2] << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> labels = SLIC_sequential(lab, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Tempo SLIC: " << ms << " ms\n";

    int unlabeled = static_cast<int>(std::count(labels.begin(), labels.end(), -1));
    std::set<int> distinct(labels.begin(), labels.end());
    int max_label = *std::max_element(labels.begin(), labels.end());

    std::cout << "K richiesto:        " << K << "\n";
    std::cout << "Superpixel trovati: " << distinct.size() << "\n";
    std::cout << "Etichetta massima:  " << max_label << "\n";
    std::cout << "Pixel senza label:  " << unlabeled << "\n";

    cv::Mat result = draw_boundaries(img, labels);
    cv::imwrite(base + "output_22090.png", result);
    std::cout << "Salvato: output_22090.png\n";

    return 0;
}