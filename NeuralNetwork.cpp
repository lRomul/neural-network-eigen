#include "NeuralNetwork.h"

#include <iostream>
#include <fstream>


NeuralNetwork::NeuralNetwork(vector<size_t> const& layerSizes, double regulParam)
	: layerSizes(layerSizes), regulParam(regulParam)
{

	if (NumberOfLayers() < 2)				// number of layer equal or more then 2 
		throw exception();

	for (size_t i = 0; i < NumberOfLayers() - 1; ++i)
	{
		size_t rows = layerSizes[i + 1];
		size_t cols = layerSizes[i] + 1;
		Weights.push_back(MatrixXd::Zero(rows, cols));
	}

	neuronalActiv.resize(NumberOfLayers());
	delta.resize(NumberOfLayers() - 1);
	WeightsGrad.resize(Weights.size());

	RandomWeights();
}

size_t NeuralNetwork::GetInputSize() const
{
	return layerSizes.front();
}

size_t NeuralNetwork::GetOutputSize() const
{
	return layerSizes.back();
}

size_t NeuralNetwork::NumberOfLayers() const
{
	return layerSizes.size();
}

vector<size_t> NeuralNetwork::GetLayerSizes() const
{
	return layerSizes;
}

double NeuralNetwork::GetRegulParam() const
{
	return regulParam;
}

void NeuralNetwork::SetRegulParam(double regulParam)
{
	this->regulParam = regulParam;
}

//Randomly initialize the weights of a layer so that break the symmetry while training the neural network.
void NeuralNetwork::RandomWeights()
{
	double eps = sqrt(6) / sqrt(GetInputSize() + GetOutputSize());

	for (size_t i = 0; i < NumberOfLayers() - 1; ++i)
	{
		size_t rows = layerSizes[i + 1];
		size_t cols = layerSizes[i] + 1;
		Weights[i] = MatrixXd::Random(rows, cols) * eps;
	}
}


void NeuralNetwork::SetWeights(vector<MatrixXd> const& weights)
{
	if (weights.size() != Weights.size())
		throw exception();
	for (size_t i = 0; i < weights.size(); ++i)
		if (weights[i].rows() != Weights[i].rows() || weights[i].cols() != Weights[i].cols())
			throw exception();

	Weights = weights;
}

vector<MatrixXd> NeuralNetwork::GetWeights() const
{
	return Weights;
}


VectorXd NeuralNetwork::Feedforward(MatrixXd const& X)
{
	size_t numberOfExamples = X.rows();

	neuronalActiv[0].resize(X.rows(), X.cols() + 1);
	neuronalActiv[0] << VectorXd::Ones(X.rows()), X;

	for (size_t i = 1; i < NumberOfLayers(); ++i)
	{
		if (i != NumberOfLayers() - 1) {
			neuronalActiv[i].resize(numberOfExamples, layerSizes[i] + 1);
			neuronalActiv[i].col(0) << VectorXd::Ones(numberOfExamples);
			neuronalActiv[i].rightCols(neuronalActiv[i].cols() - 1) = Sigmoid(neuronalActiv[i - 1] * Weights[i - 1].transpose());
		}
		else
			neuronalActiv[i] = Sigmoid(neuronalActiv[i - 1] * Weights[i - 1].transpose());
	}

	return neuronalActiv.back();
}

VectorXd NeuralNetwork::Predict(MatrixXd const& X)
{
	Feedforward(X);

	int maxInd = 0;
	VectorXd predictions(X.rows());
	for (int i = 0; i < X.rows(); ++i)
	{
		neuronalActiv.back().row(i).maxCoeff(&maxInd);
		predictions(i) = maxInd;
	}

	return predictions;
}


pair<double, vector<MatrixXd>> NeuralNetwork::CostFunction(MatrixXd const& X, MatrixXd const& Y)
{
	size_t numberOfExamples = X.rows();

	// Feedforward
	neuronalActiv[0] = X;

	for (size_t i = 1; i < NumberOfLayers(); ++i)
	{
		if (i != NumberOfLayers() - 1) {
			neuronalActiv[i].resize(numberOfExamples, layerSizes[i] + 1);
			neuronalActiv[i].col(0) << VectorXd::Ones(numberOfExamples);
			neuronalActiv[i].rightCols(neuronalActiv[i].cols() - 1) = Sigmoid(neuronalActiv[i - 1] * Weights[i - 1].transpose());
		}
		else
			neuronalActiv[i] = Sigmoid(neuronalActiv[i - 1] * Weights[i - 1].transpose());
	}

	// Cost function
	double regulSum = 0;
	for (size_t i = 0; i < NumberOfLayers() - 1; ++i)
	{
		regulSum += Weights[i].rightCols(layerSizes[i]).array().square().sum();
	}

	double costSum = (-1.0 / numberOfExamples) * (neuronalActiv.back().array().log().matrix().cwiseProduct(Y)
		+ (1.0 - neuronalActiv.back().array()).log().matrix().cwiseProduct((1-Y.array()).matrix())).sum()
			+ (regulParam / (2.0 * numberOfExamples)) * regulSum;		// cross-entropy cost function

	// Backpropagation
	delta.back() = neuronalActiv.back() - Y;

	for (int i = NumberOfLayers() - 3; i >= 0; --i)
	{
		delta[i] = (delta[i + 1] * Weights[i + 1]).cwiseProduct(SigmoidDerivative(neuronalActiv[i + 1]));
	}

	for (size_t i = 0; i < WeightsGrad.size(); ++i)
	{
		if (i != WeightsGrad.size() - 1)
			WeightsGrad[i] = (1.0 / numberOfExamples) * (delta[i].rightCols(delta[i].cols() - 1).transpose() * neuronalActiv[i]);
		else
			WeightsGrad[i] = (1.0 / numberOfExamples) * (delta[i].transpose() * neuronalActiv[i]);
		WeightsGrad[i].rightCols(WeightsGrad[i].cols() - 1) += (regulParam / numberOfExamples) * Weights[i].rightCols(Weights[i].cols() - 1);
	}

	return pair<double, vector<MatrixXd>>(costSum, WeightsGrad);
}


double NeuralNetwork::BachTrain(MatrixXd const& inputX, VectorXd const& y, size_t maxIter, double leaningRate)
{
	MatrixXd X(inputX.rows(), inputX.cols() + 1);
	X << VectorXd::Ones(X.rows()), inputX;			// add bias units

	MatrixXd Y = MatrixXd::Zero(X.rows(), layerSizes.back());
	for (int i = 0; i < Y.rows(); ++i)
		Y(i, int(y(i))) = 1;

	pair<double, vector<MatrixXd>> costSum_weightsGrad;
	for (size_t i = 0; i < maxIter; ++i)
	{
		costSum_weightsGrad = CostFunction(X, Y);

		std::cout << "Iteration: " << i + 1 << "\t" << "Cost function: " << costSum_weightsGrad.first << "\n";

		for (size_t j = 0; j < NumberOfLayers() - 1; ++j)   // so slow
			Weights[j] -= leaningRate * costSum_weightsGrad.second[j];

	}

	return costSum_weightsGrad.first;
}


double NeuralNetwork::MiniBachTrain(MatrixXd const& inputX, VectorXd const& y, size_t bachSize, size_t maxIter, double leaningRate)
{
	MatrixXd X(inputX.rows(), inputX.cols() + 1);
	X << VectorXd::Ones(X.rows()), inputX;			// add bias units

	MatrixXd Y = MatrixXd::Zero(X.rows(), layerSizes.back());
	for (int i = 0; i < Y.rows(); ++i)
		Y(i, int(y(i))) = 1;

	double costSum = 0;

	pair<double, vector<MatrixXd>> costSum_weightsGrad;
	for (size_t i = 0; i < maxIter; ++i)
	{
		costSum = 0;

		for (size_t j = 0; j < X.rows() / bachSize; ++j)
		{
			costSum_weightsGrad = CostFunction(X.block(j*bachSize, 0, bachSize, X.cols()),
				Y.block(j*bachSize, 0, bachSize, Y.cols()));

			costSum += costSum_weightsGrad.first;

			for (size_t j = 0; j < NumberOfLayers() - 1; ++j)
				Weights[j] -= leaningRate * costSum_weightsGrad.second[j];
		}

		costSum /= X.rows() / bachSize;

		std::cout << "Epoch: " << i + 1 << "\t\t" << "Cost function: " << costSum << "\n";
	}

	return costSum;
}


MatrixXd Sigmoid(MatrixXd const& x)
{
	return 1.0 / (1.0 + (-x).array().exp());
}

MatrixXd SigmoidDerivative(MatrixXd const& x)
{
	return x.cwiseProduct((1.0 - x.array()).matrix());
}
