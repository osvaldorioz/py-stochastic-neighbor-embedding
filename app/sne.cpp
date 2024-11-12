#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>

namespace py = pybind11;

class StochasticNeighborEmbedding {
public:
    StochasticNeighborEmbedding(const std::vector<std::vector<double>>& data, int output_dim = 2, double perplexity = 30.0, double learning_rate = 200.0)
        : data_(data), output_dim_(output_dim), perplexity_(perplexity), learning_rate_(learning_rate) {
        n_ = data_.size();
        if (n_ == 0) {
            throw std::runtime_error("Error: El conjunto de datos está vacío.");
        }
        init_low_dim_points();
        compute_symmetrized_high_dim_probabilities();
    }

    std::vector<std::vector<double>> fit(int num_iterations = 1000) {
        for (int iter = 0; iter < num_iterations; ++iter) {
            compute_gradient();
            update_low_dim_points();
        }
        return low_dim_points_;
    }

private:
    const std::vector<std::vector<double>>& data_;
    int n_;
    int output_dim_;
    double perplexity_;
    double learning_rate_;
    double regularization_ = 0.1; // Aumenta regularización para evitar `NaN`
    std::vector<std::vector<double>> low_dim_points_;
    std::vector<std::vector<double>> high_dim_probs_; // Symmetric P_ij
    std::vector<std::vector<double>> gradients_;

    void init_low_dim_points() {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-0.5, 0.5);
        low_dim_points_.resize(n_, std::vector<double>(output_dim_));
        gradients_.resize(n_, std::vector<double>(output_dim_, 0.0));

        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < output_dim_; ++j) {
                low_dim_points_[i][j] = distribution(generator);
            }
        }
    }

    double euclidean_distance(const std::vector<double>& p1, const std::vector<double>& p2) const {
        double sum = 0.0;
        for (size_t i = 0; i < p1.size(); ++i) {
            sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
        }
        return std::sqrt(std::max(sum, 1e-10)); // Evitar distancias cercanas a 0
    }

    // Calcula las probabilidades simétricas P_ij en alta dimensión
    void compute_symmetrized_high_dim_probabilities() {
        high_dim_probs_.resize(n_, std::vector<double>(n_, 0.0));
        
        for (int i = 0; i < n_; ++i) {
            std::vector<double> distances(n_, 0.0);

            for (int j = 0; j < n_; ++j) {
                if (i != j) {
                    distances[j] = euclidean_distance(data_[i], data_[j]);
                }
            }

            double beta = 1.0;
            double sum_p = 0.0;

            double tol = 1e-5;
            for (int loop = 0; loop < 50; ++loop) {
                sum_p = 0.0;
                for (int j = 0; j < n_; ++j) {
                    if (i != j) {
                        high_dim_probs_[i][j] = std::exp(-distances[j] * distances[j] * beta);
                        sum_p += high_dim_probs_[i][j];
                    }
                }

                if (sum_p < 1e-10) break;

                for (int j = 0; j < n_; ++j) {
                    if (i != j) {
                        high_dim_probs_[i][j] /= sum_p;
                    }
                }

                double perplexity = 0.0;
                for (int j = 0; j < n_; ++j) {
                    if (i != j && high_dim_probs_[i][j] > 0.0) {
                        perplexity += high_dim_probs_[i][j] * std::log(high_dim_probs_[i][j]);
                    }
                }
                perplexity = std::exp(-perplexity);

                if (std::abs(perplexity - perplexity_) < tol) break;

                if (perplexity > perplexity_) {
                    beta *= 2.0;
                } else {
                    beta /= 2.0;
                }
            }
        }

        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < n_; ++j) {
                if (i != j) {
                    high_dim_probs_[i][j] = (high_dim_probs_[i][j] + high_dim_probs_[j][i]) / (2.0 * n_);
                }
            }
        }
    }

    void compute_gradient() {
        for (int i = 0; i < n_; ++i) {
            std::fill(gradients_[i].begin(), gradients_[i].end(), 0.0);
        }

        for (int i = 0; i < n_; ++i) {
            for (int j = i + 1; j < n_; ++j) {
                double dist_low_dim = euclidean_distance(low_dim_points_[i], low_dim_points_[j]);
                double q_ij = std::exp(-dist_low_dim * dist_low_dim);
                q_ij = std::max(q_ij, 1e-10); // Evitar q_ij = 0

                double grad_multiplier = 4 * (high_dim_probs_[i][j] - q_ij) / (1.0 + dist_low_dim * dist_low_dim + 1e-10);

                for (int d = 0; d < output_dim_; ++d) {
                    double gradient = grad_multiplier * (low_dim_points_[i][d] - low_dim_points_[j][d]);
                    gradients_[i][d] += gradient;
                    gradients_[j][d] -= gradient;
                }
            }
        }

        for (int i = 0; i < n_; ++i) {
            for (int d = 0; d < output_dim_; ++d) {
                gradients_[i][d] += regularization_ * low_dim_points_[i][d];
            }
        }
    }

    void update_low_dim_points() {
        for (int i = 0; i < n_; ++i) {
            for (int d = 0; d < output_dim_; ++d) {
                low_dim_points_[i][d] -= learning_rate_ * gradients_[i][d];
                
                if (std::isnan(low_dim_points_[i][d]) || std::fabs(low_dim_points_[i][d]) < 1e-10) {
                    low_dim_points_[i][d] = (rand() % 100) / 500.0 - 0.1;
                }
            }
        }
    }
};

// Enlace con Pybind11
PYBIND11_MODULE(sne, m) {
    py::class_<StochasticNeighborEmbedding>(m, "StochasticNeighborEmbedding")
        .def(py::init<const std::vector<std::vector<double>>& , int, double, double>(), 
             py::arg("data"), py::arg("output_dim") = 2, py::arg("perplexity") = 30.0, py::arg("learning_rate") = 200.0)
        .def("fit", &StochasticNeighborEmbedding::fit, py::arg("num_iterations") = 1000);
}
