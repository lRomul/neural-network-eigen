#pragma once

#include <vector>
#include <utility>
#include <Eigen/Dense> // Linear algebra: MatrixXd, VectorXd

using namespace Eigen;
using namespace std;

// Feedforward neural network with Backpropagation learning
class NeuralNetwork
{
public:

	// layer_sizes - vector with sizes of layers ( eg {400, 25, 10} )
	// lambda - regularization parameter
	NeuralNetwork(vector<size_t> const& layerSizes, double regularParam = 0);

	size_t GetInputSize() const;
	size_t GetOutputSize() const;
	size_t NumberOfLayers() const;
	vector<size_t> GetLayerSizes() const;
	double GetRegulParam() const;
	void SetRegulParam(double regulParam);
	void RandomWeights();
	void SetWeights(vector<MatrixXd> const& weights);
	vector<MatrixXd> GetWeights() const;

	VectorXd Feedforward(MatrixXd const& X);

	// Outputs the predicted label of X
	VectorXd Predict(MatrixXd const& X);

	// Classic Gradient Descent
	// X, y - training examples
	// maxIter - number of iterations
	// leaningRate - rate for Gradient Descent
	double BachTrain(MatrixXd const& X, VectorXd const& y,
		size_t maxIter = 100, double leaningRate = 1);

	// Mini-Batch Gradient Descent
	// Compute the gradient for bachSize examples at each step
	double MiniBachTrain(MatrixXd const& X, VectorXd const& y,
		size_t bachSize, size_t maxIter = 100, double leaningRate = 1);

private:

	// Computes the cost and gradient of the neural network
	pair<double, vector<MatrixXd>> CostFunction(MatrixXd const& X, MatrixXd const& y);

	vector<size_t> layerSizes;
	double regulParam;
	vector<MatrixXd> Weights; // matrix of weights


	// optimization (not have to create the matrix for each iteration)
	vector<MatrixXd> neuronalActiv;
	vector<MatrixXd> delta;
	vector<MatrixXd> WeightsGrad;
	// can delete after training

};

MatrixXd Sigmoid(MatrixXd const&); // Activation function
MatrixXd SigmoidDerivative(MatrixXd const&); // Derivative of sigmoid function
