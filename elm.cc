#include "elm.h"
#include <cmath>
#include <cstring>
#include <iostream>

// Baseline https://github.com/burnpiro/elm-pure/blob/master/model.py

// -------------------- Activation --------------------

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double fourier(double x) {
	return sin(x);
}

double hardlimit(double x) {
	return (x >= 0.0) ? 1.0 : 0.0;
}

double identity(double x) {
	return x;
}


ActivationFunc getActivation(const char* name) {
	if (strcmp(name, "sigmoid") == 0) return sigmoid;
	if (strcmp(name, "fourier") == 0) return fourier;
	if (strcmp(name, "hardlimit") == 0) return hardlimit;
	return identity;
}


// -------------------- Matrix --------------------
// Source: ChaptGPT generated

// C = A(rowsA x colsA) * B(colsA x colsB)
void matMul(double* A, double* B, double* C, int rowsA, int colsA, int colsB) {
	for (int i = 0; i < rowsA; i++) {
		for (int j = 0; j < colsB; j++) {
			C[i * colsB + j] = 0.0;
			for (int k = 0; k < colsA; k++) {
				C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
			}
		}
	}
}

// B = transpose(A)
void transpose(double* A, double* B, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			B[j * rows + i] = A[i * cols + j];
		}
	}
}

// C = A + row vector b broadcast over rows
void addBias(double* A, double* b, double* C, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] + b[j];
		}
	}
}

void applyActivation(double* A, double* B, int rows, int cols, ActivationFunc func) {
	for (int i = 0; i < rows * cols; i++) {
		B[i] = func(A[i]);
	}
}

// Invert square matrix A(n x n) into inv using Gauss-Jordan
bool invertMatrix(double* A, double* inv, int n) {
	double temp[100][200]; // enough for small examples only

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			temp[i][j] = A[i * n + j];
		}
		for (int j = 0; j < n; j++) {
			temp[i][j + n] = (i == j) ? 1.0 : 0.0;
		}
	}

	for (int i = 0; i < n; i++) {
		double pivot = temp[i][i];
		if (fabs(pivot) < 1e-12) {
			return false;
		}

		for (int j = 0; j < 2 * n; j++) {
			temp[i][j] /= pivot;
		}

		for (int k = 0; k < n; k++) {
			if (k == i) continue;
			double factor = temp[k][i];
			for (int j = 0; j < 2 * n; j++) {
				temp[k][j] -= factor * temp[i][j];
			}
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			inv[i * n + j] = temp[i][j + n];
		}
	}

	return true;
}

// beta = (H^T H)^-1 H^T Y
bool computeBeta(double* H, double* Y, double* beta, int samples, int hidden, int outUnits) {
	double Ht[NUM_HIDDEN_UNITS * NUM_SAMPLES];
	double HtH[NUM_HIDDEN_UNITS * NUM_HIDDEN_UNITS];
	double HtH_inv[NUM_HIDDEN_UNITS * NUM_HIDDEN_UNITS];
	double HtY[NUM_HIDDEN_UNITS * NUM_OUT_UNITS];

	transpose(H, Ht, samples, hidden);
	matMul(Ht, H, HtH, hidden, samples, hidden);

	if (!invertMatrix(HtH, HtH_inv, hidden)) {
		return false;
	}

	matMul(Ht, Y, HtY, hidden, samples, outUnits);
	matMul(HtH_inv, HtY, beta, hidden, hidden, outUnits);

	return true;
}

// -------------------- ELM --------------------

void fitELM(
	double* X,
	double* Y,
	double* W,
	double* bias,
	double* beta,
	int samples,
	int inputNodes,
	int hiddenUnits,
	int outUnits,
	ActivationFunc activation
) {
	int size = NUM_SAMPLES * NUM_HIDDEN_UNITS;
	double XW[size];
	double XWb[size];
	double H[size];

	matMul(X, W, XW, samples, inputNodes, hiddenUnits);
	addBias(XW, bias, XWb, samples, hiddenUnits);
	applyActivation(XWb, H, samples, hiddenUnits, activation);

	if (!computeBeta(H, Y, beta, samples, hiddenUnits, outUnits)) {
		std::cout << "Failed to compute beta: matrix inversion failed.\n";
	}
}

void predictELM(
	double* X,
	double* W,
	double* bias,
	double* beta,
	double* pred,
	int samples,
	int inputNodes,
	int hiddenUnits,
	int outUnits,
	ActivationFunc activation
) {
	int size = NUM_SAMPLES * NUM_HIDDEN_UNITS;
	double XW[size];
	double XWb[size];
	double H[size];

	matMul(X, W, XW, samples, inputNodes, hiddenUnits);
	addBias(XW, bias, XWb, samples, hiddenUnits);
	applyActivation(XWb, H, samples, hiddenUnits, activation);
	matMul(H, beta, pred, samples, hiddenUnits, outUnits);
}
