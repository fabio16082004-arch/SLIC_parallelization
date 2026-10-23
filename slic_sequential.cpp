#include <iostream>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>

void rgb2lab(char R, char G, char B,
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

struct Pixel {
    int l;
    int a;
    int b;
    int x;
    int y;

    Pixel() = default;

    Pixel(int l, int a, int b, int x, int y) {
        this->l = l;
        this->a = a;
        this->b = b;
        this->x = x;
        this->y = y;
    }
};

/*STEP - 1: INITIALISING CENTERS*/
double gradient(const cv::Mat& lab, int x, int y) {
    cv::Vec3f dx = lab.at<cv::Vec3f>(y, x + 1) - lab.at<cv::Vec3f>(y, x - 1);
    cv::Vec3f dy = lab.at<cv::Vec3f>(y + 1, x) - lab.at<cv::Vec3f>(y - 1, x);
    return cv::norm(dx.dot(dx) + dy.dot(dy), cv::NORM_L2SQR);
}

void perturbe_center(const cv::Mat& lab, int &x, int &y) {
    int min_x = x;
    int min_y = y;
    double grad = gradient(lab, x, y);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int nx = x + (i - 1);
            int ny = y + (j - 1);
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

std::vector<Pixel> inizialize_centers(const cv::Mat& lab, int K) {   // lab = immagine Lab in float
    int width = lab.cols, height = lab.rows;
    int N = width * height;
    int S = static_cast<int>(std::sqrt(static_cast<double>(N) / K) + 0.5);

    int cols = static_cast<int>(static_cast<double>(width)  / S + 0.5);
    int rows = static_cast<int>(static_cast<double>(height) / S + 0.5);

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



int main() {
    int K = 196;
    struct Pixel centers[K];

    std::string dataset_directory = "archive/images/train";

    for (const auto& d: std::filesystem::directory_iterator(dataset_directory)) {
        std::string file_name = d.path().string();
        cv::Mat img = cv::imread(file_name, cv::IMREAD_COLOR);
    }





    cv::Mat img = cv::imread("/home/fbi168/CLionProjects/SLIC/archive/images/train/22090.jpg");   // carica in BGR, 8 bit
    if (img.empty()) {
        std::cerr << "Immagine non trovata\n";
        return 1;
    }

    std::cout << "Dimensioni: " << img.cols << " x " << img.rows << "\n";

    int x = 10, y = 20;
    cv::Vec3b p = img.at<cv::Vec3b>(y, x);   // attenzione: prima la riga (y), poi la colonna (x)
    int b = p[0], g = p[1], r = p[2];        // l'ordine è B, G, R, non R, G, B
    std::cout << "R=" << r << " G=" << g << " B=" << b << "\n";
}