#ifndef ELM_H
#define ELM_H

// Baseline https://github.com/burnpiro/elm-pure/blob/master/model.py

const int NUM_SAMPLES = 64; // must be bigger than NUM_HIDDEN_UNITS
const int NUM_INPUT_NODES = 128;
const int NUM_HIDDEN_UNITS = 16;
const int NUM_OUT_UNITS = 1;

typedef double (*ActivationFunc)(double);

double sigmoid(double x);
double fourier(double x);
double hardlimit(double x);
double identity(double x);

ActivationFunc getActivation(const char* name);

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
);

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
);

#endif
