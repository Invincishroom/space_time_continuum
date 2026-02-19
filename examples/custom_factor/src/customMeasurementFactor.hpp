'''
This is an example factor to demonstrate expected contribution formats. 
It adds a regularizing error on the strain component of the state, 
encouraging the estimator to find solutions with zero strain. 

It does not require any specific hardware, nor does it model any specific sensor. 
It simply demonstrates how to create a custom measurement factor by inheriting from 
the MeasurementFactor base class and implementing the getError and getJacobian functions.

The author of this factor is Spencer Teetaert, spencer.teetaert@robotics.utias.utoronto.ca

If you use this code in your project, please cite our paper:
@article{exampleCitation
    title={Spacetime Factor: An Example Paper for an Example Factor},
    author={Spencer Teetaert},
    journal={The Journal of Non-Existant Research},
    year={2026}
}
'''


#ifndef CUSTOM_MEASUREMENT_FACTOR_HPP
#define CUSTOM_MEASUREMENT_FACTOR_HPP

#include <Eigen/Dense>

#include "spacetime/factors/MeasurementFactor.hpp"
#include "spacetime/utilities.hpp"
#include "spacetime/types.hpp"

using Node = Spacetime::SystemState<DTYPE>::Node;

class CustomMeasurementFactor : public Spacetime::Factors::MeasurementFactor
{

public:
    CustomMeasurementFactor(Eigen::Matrix<double, 6, 6> weight, const Spacetime::SensorMeasurement &meas)
        : Spacetime::Factors::MeasurementFactor(weight, meas){}

    ~CustomMeasurementFactor() = default;

    Eigen::VectorX<DTYPE> getError(const std::vector<Node> &nodes) const override
    {
        const int ERROR_SIZE = 6;
        const int NUM_NODES = NUM_NODES; // This factor operates on a single node. Adjust as needed for factors that operate on multiple nodes.

        // Check that ensures the correct number of nodes are passed to the factor. For measurements, this is typically 1. 
        assert(nodes.size() == 1); 

        // Define the error. This can be any size you need. Here, we are defining a regularizing error on the strain component of the state. 
        Eigen::Vector<DTYPE, ERROR_SIZE> e;
        e << nodes[0].epsilon;

        // This applies the mask to the error, zeroing out any components that are not valid. This mask is mutable, 
        // allowing the factor to modify it at runtime if needed (e.g. an ICP factor may reject some points and update the mask accordingly).
        for (unsigned int i = 0; i < ERROR_SIZE; i++)
        {
            if (this->m_meas.mask(i, 0) == 0)
            {
                e(i, 0) = 0;
            }
        }
        return e;
    };

    Eigen::MatrixX<double> getJacobian(const std::vector<Node> &nodes, Eigen::VectorX<DTYPE> &e) const override
    {
        const int ERROR_SIZE = 6;
        const int NUM_NODES = 1; // This factor operates on a single node. Adjust as needed for factors that operate on multiple nodes.

        // Check that ensures the correct number of nodes are passed to the factor. For measurements, this is typically 1. 
        assert(nodes.size() == NUM_NODES);

        Eigen::Matrix<double, ERROR_SIZE, 18 * NUM_NODES> H = Eigen::Matrix<double, ERROR_SIZE, 18 * NUM_NODES>::Zero();

        // Define the error jacobian. Here, we are defining a regularizing error on the strain component of the state. 
        // The Jacobian has 18 columns per node, corresponding to the 18 state variables (pose, strain, varpi) of each node. 
        H.block<ERROR_SIZE, ERROR_SIZE>(0, 6) = Eigen::Matrix<double, ERROR_SIZE, ERROR_SIZE>::Identity();
        e = getError(nodes);

        // This applies the mask to the error, zeroing out any components that are not valid. This mask is mutable, 
        // allowing the factor to modify it at runtime if needed (e.g. an ICP factor may reject some points and update the mask accordingly).
        for (unsigned int i = 0; i < ERROR_SIZE; i++)
        {
            if (this->m_meas.mask(i, 0) == 0)
            {
                H.row(i).setZero();
            }
        }
        return H;
    };
};
#endif // CUSTOM_MEASUREMENT_FACTOR_HPP