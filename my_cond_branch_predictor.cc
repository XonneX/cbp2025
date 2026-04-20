#include "my_cond_branch_predictor.h"
#include "elm.h"
#include <iostream>
#include <random>

static double W[NUM_INPUT_NODES * NUM_HIDDEN_UNITS];
static double b[NUM_HIDDEN_UNITS];
static double beta[NUM_HIDDEN_UNITS * NUM_OUT_UNITS];

static double Xbuf[NUM_SAMPLES * NUM_INPUT_NODES];
static double Ybuf[NUM_SAMPLES * NUM_OUT_UNITS];
static int train_count = 0;
static int train_pos = 0;

void SampleCondPredictor::setup()
{
	active_hist.ghist = 0;
	active_hist.tage_pred = false;

	std::mt19937 rng(12345);
	std::uniform_real_distribution<double> w_dist(-1.0, 1.0);
	std::uniform_int_distribution<int> bias_dist(-4, 4);

	for (int i = 0; i < NUM_INPUT_NODES; ++i) {
		for (int h = 0; h < NUM_HIDDEN_UNITS; ++h) {
			W[i * NUM_HIDDEN_UNITS + h] = w_dist(rng);
		}
	}

	for (int h = 0; h < NUM_HIDDEN_UNITS; ++h) {
		b[h] = static_cast<double>(bias_dist(rng));
		beta[h] = 0.0;
	}

	//std::memset(Xbuf, 0, sizeof(Xbuf));
	//std::memset(Ybuf, 0, sizeof(Ybuf));
	train_count = 0;
	train_pos = 0;
}

void SampleCondPredictor::terminate()
{
}

void build_features(uint64_t PC, const SampleHist& hist, double x[NUM_INPUT_NODES])
{
	int offset = 0;

	for (int i = 0; i < 64; ++i) {
		x[offset++] = ((PC >> i) & 1ULL) ? 1.0 : 0.0;
	}

	for (int i = 0; i < 64; ++i) {
		x[offset++] = ((hist.ghist >> i) & 1ULL) ? 1.0 : 0.0;
	}

	// padding at the end (if needed)
	while (offset < NUM_INPUT_NODES) {
		x[offset++] = 0.0;
	}
}

bool SampleCondPredictor::predict_using_given_hist(
	uint64_t seq_no,
	uint8_t piece,
	uint64_t PC,
	const SampleHist& hist_to_use,
	const bool pred_time_predict
) {
	double x[NUM_INPUT_NODES];
	double pred[NUM_OUT_UNITS];

	build_features(PC, hist_to_use, x);

	predictELM(
		x,
		W,
		b,
		beta,
		pred,
		1,
		NUM_INPUT_NODES,
		NUM_HIDDEN_UNITS,
		NUM_OUT_UNITS,
		getActivation("sigmoid")
	);

	return pred[0] >= 0.0;
}

void SampleCondPredictor::update(
	uint64_t PC,
	bool resolveDir,
	bool pred_taken,
	uint64_t nextPC,
	const SampleHist& hist_to_use
) {
	double x[NUM_INPUT_NODES];
	build_features(PC, hist_to_use, x);
	
	double target = resolveDir ? 1.0 : 0.0;

	for (int i = 0; i < NUM_INPUT_NODES; ++i) {
		Xbuf[train_pos * NUM_INPUT_NODES + i] = x[i];
	}
	Ybuf[train_pos * NUM_OUT_UNITS] = target;

	train_pos = (train_pos + 1) % NUM_SAMPLES;
	if (train_count < NUM_SAMPLES) {
		++train_count;
	}

	if (train_count >= NUM_SAMPLES && pred_taken != resolveDir) {
		fitELM(
			Xbuf,
			Ybuf,
			W,
			b,
			beta,
			NUM_SAMPLES,
			NUM_INPUT_NODES,
			NUM_HIDDEN_UNITS,
			NUM_OUT_UNITS,
			getActivation("sigmoid")
		);
	}
}