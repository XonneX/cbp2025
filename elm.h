// Baseline https://github.com/burnpiro/elm-pure/blob/master/model.py

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

const int NUM_SAMPLES = 64; // must be bigger than NUM_HIDDEN_UNITS
const int NUM_INPUT_NODES = 128;
const int NUM_HIDDEN_UNITS = 16;
const int NUM_OUT_UNITS = 1;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

enum class Activation {
	Sigmoid,
	Fourier,
	Hardlimit,
	Identity
};

enum class Loss {
	MSE,
	MAE
};

inline Activation getActivation(const std::string& name) {
	if (name == "sigmoid") return Activation::Sigmoid;
	if (name == "fourier") return Activation::Fourier;
	if (name == "hardlimit") return Activation::Hardlimit;
	if (name == "identity") return Activation::Identity;

	throw std::invalid_argument("Unknown activation: " + name);
}

inline Loss getLoss(const std::string& name) {
	if (name == "mse") return Loss::MSE;
	if (name == "mae") return Loss::MAE;

	throw std::invalid_argument("Unknown loss: " + name);
}

inline Matrix applyActivation(const Matrix& x, Activation activation) {
	switch (activation) {
		case Activation::Sigmoid:
			return (1.0 / (1.0 + (-x.array()).exp())).matrix();

		case Activation::Fourier:
			return x.array().sin().matrix();

		case Activation::Hardlimit:
			return (x.array() >= 0.0).cast<double>().matrix();

		case Activation::Identity:
			return x;
	}

	return x;
}

template<typename MatrixType>
MatrixType pseudoInverse(
	const MatrixType& a,
	double epsilon = std::numeric_limits<double>::epsilon()
) {
	Eigen::JacobiSVD<MatrixType> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);

	const double tolerance =
		epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs()(0);

	return svd.matrixV()
		* (svd.singularValues().array().abs() > tolerance)
			.select(svd.singularValues().array().inverse(), 0)
			.matrix()
			.asDiagonal()
		* svd.matrixU().adjoint();
}

inline double meanSquaredError(const Matrix& y, const Matrix& pred) {
	return 0.5 * (y - pred).array().square().mean();
}

inline double meanAbsError(const Matrix& y, const Matrix& pred) {
	return (y - pred).array().abs().mean();
}

inline double computeLoss(const Matrix& y, const Matrix& pred, Loss loss) {
	switch (loss) {
		case Loss::MSE:
			return meanSquaredError(y, pred);

		case Loss::MAE:
			return meanAbsError(y, pred);
	}

	return meanSquaredError(y, pred);
}

inline int argmaxRow(const Matrix& m, int row) {
	Eigen::Index index;
	m.row(row).maxCoeff(&index);
	return static_cast<int>(index);
}

class ELM {
public:
	ELM(
		int numInputNodes,
		int numHiddenUnits,
		int numOutUnits,
		const std::string& activation = "sigmoid",
		const std::string& loss = "mse"
	)
		: numInputNodes(numInputNodes),
		  numHiddenUnits(numHiddenUnits),
		  numOutUnits(numOutUnits),
		  activation(getActivation(activation)),
		  loss(getLoss(loss)),
		  beta(randomMatrix(numHiddenUnits, numOutUnits)),
		  w(randomMatrix(numInputNodes, numHiddenUnits)),
		  bias(Vector::Zero(numHiddenUnits))
	{
		printShapes();
	}

	ELM(
		int numInputNodes,
		int numHiddenUnits,
		int numOutUnits,
		const std::string& activation,
		const std::string& loss,
		const Matrix& betaInit,
		const Matrix& wInit,
		const Vector& biasInit
	)
		: numInputNodes(numInputNodes),
		  numHiddenUnits(numHiddenUnits),
		  numOutUnits(numOutUnits),
		  activation(getActivation(activation)),
		  loss(getLoss(loss)),
		  beta(betaInit),
		  w(wInit),
		  bias(biasInit)
	{
		validateShapes();
		printShapes();
	}

	void fit(const Matrix& X, const Matrix& Y, bool displayTime = false) {
		Matrix H = hiddenLayerOutput(X);

		if (displayTime) {
			const auto start = std::chrono::high_resolution_clock::now();

			beta = pseudoInverse(H) * Y;

			const auto stop = std::chrono::high_resolution_clock::now();
			const std::chrono::duration<double> elapsed = stop - start;

			std::cout << "Train time: " << elapsed.count() << "\n";
		} else {
			beta = pseudoInverse(H) * Y;
		}
	}

	Matrix operator()(const Matrix& X) const {
		Matrix H = hiddenLayerOutput(X);
		return H * beta;
	}

	std::pair<double, double> evaluate(const Matrix& X, const Matrix& Y) const {
		Matrix pred = (*this)(X);

		double lossValue = computeLoss(Y, pred, loss);

		int correct = 0;
		for (int i = 0; i < Y.rows(); i++) {
			if (argmaxRow(pred, i) == argmaxRow(Y, i)) {
				correct++;
			}
		}

		double acc = static_cast<double>(correct) / static_cast<double>(Y.rows());

		return {lossValue, acc};
	}

	const Matrix& getBeta() const {
		return beta;
	}

	const Matrix& getW() const {
		return w;
	}

	const Vector& getBias() const {
		return bias;
	}

private:
	int numInputNodes;
	int numHiddenUnits;
	int numOutUnits;

	Activation activation;
	Loss loss;

	Matrix beta;
	Matrix w;
	Vector bias;

	Matrix hiddenLayerOutput(const Matrix& X) const {
		Matrix XWb = (X * w).rowwise() + bias.transpose();
		return applyActivation(XWb, activation);
	}

	static Matrix randomMatrix(int rows, int cols) {
		static std::mt19937 rng(std::random_device{}());
		static std::uniform_real_distribution<double> dist(-1.0, 1.0);

		return Matrix::NullaryExpr(rows, cols, []() {
			return dist(rng);
		});
	}

	void validateShapes() const {
		if (w.rows() != numInputNodes || w.cols() != numHiddenUnits) {
			throw std::invalid_argument("Invalid W shape");
		}

		if (beta.rows() != numHiddenUnits || beta.cols() != numOutUnits) {
			throw std::invalid_argument("Invalid beta shape");
		}

		if (bias.size() != numHiddenUnits) {
			throw std::invalid_argument("Invalid bias shape");
		}
	}

	void printShapes() const {
		std::cout << "Bias shape: (" << bias.size() << ")\n";
		std::cout << "W shape: (" << w.rows() << ", " << w.cols() << ")\n";
		std::cout << "Beta shape: (" << beta.rows() << ", " << beta.cols() << ")\n";
	}
};