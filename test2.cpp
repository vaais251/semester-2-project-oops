#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include <ctime>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class NeuralNetwork {
private:
    std::vector<int> layers;
    double learning_rate;
    std::vector<MatrixXd> weights;
    std::vector<VectorXd> biases;
    std::vector<std::pair<MatrixXd, MatrixXd>> caches; // Stores (A_prev, W) for linear, Z for activation
    std::vector<MatrixXd> grads_W;
    std::vector<VectorXd> grads_b;

    MatrixXd sigmoid(const MatrixXd& z) {
        return 1.0 / (1.0 + (-z.array()).exp());
    }

    MatrixXd sigmoid_derivative(const MatrixXd& A) {
        return A.array() * (1.0 - A.array());
    }

    void initialize_parameters(int input_size) {
        std::vector<int> all_layers = {input_size};
        all_layers.insert(all_layers.end(), layers.begin(), layers.end());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.1);

        for (size_t i = 1; i < all_layers.size(); ++i) {
            MatrixXd W(all_layers[i], all_layers[i-1]);
            for (int r = 0; r < W.rows(); ++r)
                for (int c = 0; c < W.cols(); ++c)
                    W(r, c) = d(gen);
            weights.push_back(W);
            biases.push_back(VectorXd::Zero(all_layers[i]));
        }
    }

    std::pair<MatrixXd, std::pair<MatrixXd, MatrixXd>> forward_linear(const MatrixXd& A, const MatrixXd& W, const VectorXd& b) {
        MatrixXd Z = W * A + b.replicate(1, A.cols());
        return {Z, {A, W}};
    }

    std::pair<MatrixXd, MatrixXd> forward_activation(const MatrixXd& Z) {
        MatrixXd A = sigmoid(Z);
        return {A, Z};
    }

    MatrixXd forward_propagation(const MatrixXd& X) {
        caches.clear();
        MatrixXd A = X;
        for (size_t i = 0; i < weights.size(); ++i) {
            auto [Z, linear_cache] = forward_linear(A, weights[i], biases[i]);
            auto [A_next, act_cache] = forward_activation(Z);
            caches.push_back({linear_cache, act_cache});
            A = A_next;
        }
        return A;
    }

    std::tuple<MatrixXd, MatrixXd, VectorXd> backward_linear(const MatrixXd& a_prev, const MatrixXd& W, const MatrixXd& dz_l, const MatrixXd& da) {
        MatrixXd da_l = da.array() * dz_l.array();
        MatrixXd dW = da_l * a_prev.transpose();
        VectorXd db = da_l.rowwise().sum();
        MatrixXd dz = W.transpose() * da_l;
        return {dz, dW, db};
    }

    void backward_propagation(const MatrixXd& X, const MatrixXd& Y, const MatrixXd& y_hat) {
        grads_W.clear();
        grads_b.clear();
        MatrixXd dz_l = y_hat - Y;
        for (int i = weights.size() - 1; i >= 0; --i) {
            auto& [linear_cache, act_cache] = caches[i];
            MatrixXd A_prev = linear_cache.first;
            MatrixXd W = linear_cache.second;
            MatrixXd Z = act_cache;
            MatrixXd A = sigmoid(Z);
            MatrixXd da = sigmoid_derivative(A);

            auto [dz, dW, db] = backward_linear(A_prev, W, dz_l, da);
            grads_W.push_back(dW);
            grads_b.push_back(db);
            dz_l = dz;
        }
        std::reverse(grads_W.begin(), grads_W.end());
        std::reverse(grads_b.begin(), grads_b.end());
    }

    void update_parameters() {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learning_rate * grads_W[i];
            biases[i] -= learning_rate * grads_b[i];
        }
    }

    double compute_loss(const MatrixXd& y, const MatrixXd& predict) {
        const double epsilon = 1e-15;
        MatrixXd clipped = predict.array().max(epsilon).min(1.0 - epsilon);
        return -(y.array() * clipped.array().log() + (1.0 - y.array()) * (1.0 - clipped.array()).log()).mean();
    }

    double compute_accuracy(const MatrixXd& y, const MatrixXd& predict) {
        MatrixXd predictions = (predict.array() > 0.5).cast<double>();
        return (predictions.array() == y.array()).cast<double>().mean();
    }

public:
    NeuralNetwork(const std::vector<int>& layers, double learning_rate = 0.001)
        : layers(layers), learning_rate(learning_rate) {}

    void train(const MatrixXd& X, const MatrixXd& y, int epochs = 1000) {
        initialize_parameters(X.rows());
        for (int i = 0; i < epochs; ++i) {
            MatrixXd y_hat = forward_propagation(X);
            backward_propagation(X, y, y_hat);
            update_parameters();

            if (i % 100 == 0 || i == epochs - 1) {
                double loss = compute_loss(y, y_hat);
                double accuracy = compute_accuracy(y, y_hat);
                std::cout << "Epoch " << i << " -> Loss: " << loss << ", Accuracy: " << accuracy << std::endl;
            }
        }

        MatrixXd y_hat = forward_propagation(X);
        double final_loss = compute_loss(y, y_hat);
        double final_accuracy = compute_accuracy(y, y_hat);
        std::cout << "Final Results -> Loss: " << final_loss << ", Accuracy: " << final_accuracy << std::endl;
    }
};

// Function to generate make_moons-like dataset
std::pair<MatrixXd, MatrixXd> make_moons(int n_samples, double noise, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dist(0, 1);
    std::normal_distribution<> noise_dist(0, noise);

    MatrixXd X(2, n_samples);
    MatrixXd y(1, n_samples);

    int half_n = n_samples / 2;
    for (int i = 0; i < half_n; ++i) {
        double theta = dist(gen) * M_PI;
        X(0, i) = cos(theta) + noise_dist(gen);
        X(1, i) = sin(theta) + noise_dist(gen);
        y(0, i) = 0;
    }
    for (int i = half_n; i < n_samples; ++i) {
        double theta = dist(gen) * M_PI;
        X(0, i) = 1 - cos(theta) + noise_dist(gen);
        X(1, i) = 0.5 - sin(theta) + noise_dist(gen);
        y(0, i) = 1;
    }

    return {X, y};
}

int main() {
    // Generate make_moons dataset
    auto [X, y] = make_moons(1000, 0.1, 42);

    // Initialize and train the neural network
    NeuralNetwork nn({10, 12, 1}, 0.01);
    nn.train(X, y, 1000);

    return 0;
}

