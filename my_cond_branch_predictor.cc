#include "my_cond_branch_predictor.h"
#include "elm.h"
#include <iostream>
#include <random>

static ELM* elm = nullptr;

static Eigen::MatrixXd Xbuf;
static Eigen::MatrixXd Ybuf;

static int train_count = 0;
static int train_pos = 0;

void SampleCondPredictor::setup()
{
	// std::cout << "TEST 123!" << std::endl;
	active_hist.ghist = 0;
	active_hist.tage_pred = false;

	delete elm;

	elm = new ELM(
		NUM_INPUT_NODES,
		NUM_HIDDEN_UNITS,
		NUM_OUT_UNITS,
		"fourier",
		"mse"
	);

	Xbuf = Eigen::MatrixXd::Zero(NUM_SAMPLES, NUM_INPUT_NODES);
	Ybuf = Eigen::MatrixXd::Zero(NUM_SAMPLES, NUM_OUT_UNITS);

	train_count = 0;
	train_pos = 0;
}

void SampleCondPredictor::terminate()
{
	delete elm;
	elm = nullptr;
}

void build_features(uint64_t PC, const SampleHist& hist, Eigen::VectorXd& x)
{
	x.setZero();

	int offset = 0;

	for (int i = 0; i < 64; ++i) {
		x(offset++) = ((PC >> i) & 1ULL) ? 1.0 : 0.0;
	}

	for (int i = 0; i < 64; ++i) {
		x(offset++) = ((hist.ghist >> i) & 1ULL) ? 1.0 : 0.0;
	}

	// padding at the end (if needed)
	while (offset < NUM_INPUT_NODES) {
		x(offset++) = 0.0;
	}
}

bool SampleCondPredictor::predict_using_given_hist(
	uint64_t seq_no,
	uint8_t piece,
	uint64_t PC,
	const SampleHist& hist_to_use,
	const bool pred_time_predict
) {
	Eigen::VectorXd x(NUM_INPUT_NODES);
	build_features(PC, hist_to_use, x);

	Eigen::MatrixXd X(1, NUM_INPUT_NODES);
	X.row(0) = x.transpose();

	Eigen::MatrixXd pred = (*elm)(X);

	return pred(0, 0) >= 0.0;
}

void SampleCondPredictor::update(
	uint64_t PC,
	bool resolveDir,
	bool pred_taken,
	uint64_t nextPC,
	const SampleHist& hist_to_use
) {
	Eigen::VectorXd x(NUM_INPUT_NODES);
	build_features(PC, hist_to_use, x);

	Xbuf.row(train_pos) = x.transpose();
	Ybuf(train_pos, 0) = resolveDir ? 1.0 : -1.0;

	train_pos = (train_pos + 1) % NUM_SAMPLES;
		
	if (train_count < NUM_SAMPLES) {
		++train_count;
	}

	if (train_count >= NUM_SAMPLES && train_pos == 0) {
		elm->fit(Xbuf, Ybuf, false);
	}
}
