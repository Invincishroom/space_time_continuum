#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <vector>

namespace spacetime {

class OscillationDetector {
public:
    std::vector<int> observe(const Eigen::VectorXd& step, const std::vector<int>& eligible)
    {
        if (history_.size() != static_cast<std::size_t>(step.size()))
        {
            history_.assign(step.size(), {});
            cooldown_.assign(step.size(), 0);
            detected_step_magnitude_.assign(step.size(), 0.0);
        }
        std::vector<int> detected;
        for (int dimension = 0; dimension < step.size(); ++dimension)
        {
            auto& history = history_[dimension];
            if (!eligible[dimension] || !std::isfinite(step(dimension)))
            {
                history.clear();
            }
            if (cooldown_[dimension] > 0)
            {
                --cooldown_[dimension];
                continue;
            }
            if (!eligible[dimension] || !std::isfinite(step(dimension)))
            {
                continue;
            }
            history.push_back(step(dimension));
            if (history.size() > 10) history.erase(history.begin());
            if (history.size() < 10) continue;
            int reversals = 0;
            double early_magnitude = 0.0;
            double late_magnitude = 0.0;
            bool appreciable = true;
            std::vector<double> cancellation;
            for (int index = 0; index < 10; ++index)
            {
                const double magnitude = std::abs(history[index]);
                if (magnitude < 1e-6)
                {
                    appreciable = false;
                    break;
                }
                (index < 5 ? early_magnitude : late_magnitude) += magnitude;
                if (index > 0)
                {
                    const double previous = history[index - 1];
                    reversals += std::signbit(previous) != std::signbit(history[index]) ? 1 : 0;
                    cancellation.push_back(std::abs(previous + history[index]) / (std::abs(previous) + magnitude));
                }
            }
            if (!appreciable) continue;
            std::sort(cancellation.begin(), cancellation.end());
            if (reversals < 8 || cancellation[4] >= 0.15 || late_magnitude < 0.8 * early_magnitude) continue;
            detected_step_magnitude_[dimension] = std::abs(history.back());
            history.clear();
            cooldown_[dimension] = 15;
            detected.push_back(dimension);
        }
        return detected;
    }

    double reducedLearningRate(int dimension, double learning_rate, double direction_magnitude) const
    {
        return direction_magnitude > 0.0
            ? std::min(0.5 * learning_rate, 0.5 * detected_step_magnitude_[dimension] / direction_magnitude)
            : 0.5 * learning_rate;
    }

private:
    std::vector<std::vector<double>> history_;
    std::vector<int> cooldown_;
    std::vector<double> detected_step_magnitude_;
};

}