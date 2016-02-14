#include "NeuralNetwork.h"

#include <iostream>
#include <fstream>


NeuralNetwork::NeuralNetwork(vector<size_t> const& layer_sizes, double lambda)
	: layerSizes(layer_sizes), lambda(lambda)
{

	if (NumberOfLayers() < 2)				// number of layer equal or more then 2 
		throw exception();

	for (size_t i = 0; i < NumberOfLayers() - 1; ++i)
	{
		size_t rows = layerSizes[i + 1];
		size_t cols = layerSizes[i] + 1;
		Theta.push_back(MatrixXd::Zero(rows, cols));
	}

	a.resize(NumberOfLayers());
	delta.resize(NumberOfLayers() - 1);
	ThetaGrad.resize(Theta.size());

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
	return lambda;
}

void NeuralNetwork::SetRegulParam(double lambda)
{
	this->lambda = lambda;
}

//Randomly initialize the weights of a layer so that break the symmetry while training the neural network.
void NeuralNetwork::RandomWeights()
{
	double eps = sqrt(6) / sqrt(GetInputSize() + GetOutputSize());

	for (size_t i = 0; i < NumberOfLayers() - 1; ++i)
	{
		size_t rows = layerSizes[i + 1];
		size_t cols = layerSizes[i] + 1;
		Theta[i] = MatrixXd::Random(rows, cols) * eps;
	}
}

MatrixXd NeuralNetwork::Feedforward(MatrixXd const& X)
{
	size_t m = X.rows();

	a[0].resize(X.rows(), X.cols() + 1);
	a[0] << VectorXd::Ones(X.rows()), X;

	for (size_t i = 1; i < NumberOfLayers(); ++i)
	{
		if (i != NumberOfLayers() - 1) {
			a[i].resize(m, layerSizes[i] + 1);
			a[i].col(0) << VectorXd::Ones(m);
			a[i].rightCols(a[i].cols() - 1) = Sigmoid(a[i - 1] * Theta[i - 1].transpose());
		}
		else
			a[i] = Sigmoid(a[i - 1] * Theta[i - 1].transpose());
	}

	return a.back();
}

VectorXd NeuralNetwork::Predict(MatrixXd const& X)
{
	Feedforward(X);

	int maxInd = 0;
	VectorXd result(X.rows());
	for (int i = 0; i < X.rows(); ++i)
	{
		a.back().row(i).maxCoeff(&maxInd);
		result(i) = maxInd;
	}

	return result;
}


pair<double, vector<MatrixXd>> NeuralNetwork::CostFunction(MatrixXd const& X, MatrixXd const& Y)
{
	size_t m = X.rows();

	// Feedforward
	a[0] = X;

	for (size_t i = 1; i < NumberOfLayers(); ++i)
	{
		if (i != NumberOfLayers() - 1) {
			a[i].resize(m, layerSizes[i]+1);
			a[i].col(0) << VectorXd::Ones(m);
			a[i].rightCols(a[i].cols() - 1) = Sigmoid(a[i - 1] * Theta[i - 1].transpose());
		}
		else
			a[i] = Sigmoid(a[i - 1] * Theta[i - 1].transpose());
	}

	// Cost function
	double regul = 0;
	for (size_t i = 0; i < NumberOfLayers() - 1; ++i)
	{
		regul += Theta[i].rightCols(layerSizes[i]).array().square().sum();
	}

	double J = (-1.0 / m) * (a.back().array().log().matrix().cwiseProduct(Y)
		+ (1 - a.back().array()).log().matrix().cwiseProduct((1-Y.array()).matrix())).sum() + (lambda / (2.0 * m)) * regul;

	// Backpropagation
	delta.back() = a.back() - Y;

	for (int i = NumberOfLayers() - 3; i >= 0; --i)
	{
		delta[i] = (delta[i + 1] * Theta[i + 1]).cwiseProduct(a[i + 1]).cwiseProduct((1 - a[i + 1].array()).matrix());
	}

	for (size_t i = 0; i < ThetaGrad.size(); ++i)
	{
		if (i != ThetaGrad.size() - 1)
			ThetaGrad[i] = (1.0 / m) * (delta[i].rightCols(delta[i].cols() - 1).transpose() * a[i]);
		else
			ThetaGrad[i] = (1.0 / m) * (delta[i].transpose() * a[i]);
		ThetaGrad[i].rightCols(ThetaGrad[i].cols() - 1) += (lambda / m) * Theta[i].rightCols(Theta[i].cols() - 1);
	}

	return pair<double, vector<MatrixXd>>(J, ThetaGrad);
}


MatrixXd Sigmoid(MatrixXd const& m)
{
	return 1.0 / (1.0 + (-m).array().exp());
}

void NeuralNetwork::SetWeights(vector<MatrixXd> const& theta)
{
	if (theta.size() != Theta.size())
		throw exception();
	for (size_t i = 0; i < theta.size(); ++i)
		if (theta[i].rows() != Theta[i].rows() || theta[i].cols() != Theta[i].cols())
			throw exception();

	Theta = theta;
}

vector<MatrixXd> NeuralNetwork::GetWeights() const
{
	return Theta;
}


double NeuralNetwork::BachTrain(MatrixXd const& inputX, VectorXd const& y, size_t maxIter, double leaningRate)
{
	MatrixXd X(inputX.rows(), inputX.cols() + 1);
	X << VectorXd::Ones(X.rows()), inputX;			// add bias units

	MatrixXd Y = MatrixXd::Zero(X.rows(), layerSizes.back());
	for (int i = 0; i < Y.rows(); ++i)
		Y(i, int(y(i))) = 1;

	pair<double, vector<MatrixXd>> J_thetaGrad;
	for (size_t i = 0; i < maxIter; ++i)
	{
		J_thetaGrad = CostFunction(X, Y);

		std::cout << "Iteration: " << i + 1 << "\t" << "Cost function: " << J_thetaGrad.first << "\n";

		for (size_t j = 0; j < NumberOfLayers() - 1; ++j)   // so slow
			Theta[j] -= leaningRate * J_thetaGrad.second[j];

	}

	return J_thetaGrad.first;
}


double NeuralNetwork::MiniBachTrain(MatrixXd const& inputX, VectorXd const& y, size_t bachSize, size_t maxIter, double leaningRate)
{
	MatrixXd X(inputX.rows(), inputX.cols() + 1);
	X << VectorXd::Ones(X.rows()), inputX;			// add bias units

	MatrixXd Y = MatrixXd::Zero(X.rows(), layerSizes.back());
	for (int i = 0; i < Y.rows(); ++i)
		Y(i, int(y(i))) = 1;

	double J = 0;

	pair<double, vector<MatrixXd>> J_thetaGrad;
	for (size_t i = 0; i < maxIter; ++i)
	{
		J = 0;

		for (size_t j = 0; j < X.rows() / bachSize; ++j)
		{
			J_thetaGrad = CostFunction(X.block(j*bachSize, 0, bachSize, X.cols()),
				Y.block(j*bachSize, 0, bachSize, Y.cols()));

			J += J_thetaGrad.first;

			for (size_t j = 0; j < NumberOfLayers() - 1; ++j)
				Theta[j] -= leaningRate * J_thetaGrad.second[j];
		}

		J /= X.rows() / bachSize;

		std::cout << "Epoch: " << i + 1 << "\t\t" << "Cost function: " << J << "\n";
	}

	return J;
}
