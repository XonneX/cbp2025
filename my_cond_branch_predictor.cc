#include "my_cond_branch_predictor.h"
#include <iostream>
#include <random>

static constexpr int N = 32; // input size
// TODO: play with this
static constexpr int L = 64; // neurons (hidden layer)

int8_t W[N][L]; // weights hidden layer (fixed)
int16_t b[L]; // biases hidden layer (fixed)

int16_t beta[N]; // output weights

void SampleCondPredictor::setup()
{
    active_hist.ghist = 0;
    active_hist.tage_pred = false;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> bit_dist(0, 1);
    std::uniform_int_distribution<int> bias_dist(-4, 4);

    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < N; ++i) {
            W[h][i] = bit_dist(rng) ? 1 : -1;
        }
        b[h] = bias_dist(rng);
        beta[h] = 0;
    }
}

void SampleCondPredictor::terminate()
{
}

void build_features(uint64_t PC, const SampleHist& hist, int8_t x[N])
{
    int offset = 0;

    // TODO: Why 1 and -1 instead of 1 and 0?

    for (int i = 0; i < 16; ++i) {
        x[offset++] = ((PC >> i) & 1ULL) ? 1 : -1;
    }

    for (int i = 0; i < 15; ++i) {
        x[offset++] = ((hist.ghist >> i) & 1ULL) ? 1 : -1;
    }

    // x[offset++] = hist.tage_pred ? 1 : -1;
}

void hidden_layer(int8_t W[N][L], int8_t X[N], int8_t b[L]) {
    int H[N];
    for (int Ni = 0; Ni < N; ++Ni) {
        int sum = 0;
        for (int Li = 0; Li < L; ++Li) {
            H[Ni] = g(W[Ni][Li] * X[Ni] + b[Li]);
        }
    }
}

int8_t[] output_layer(int8_t H[N], int8_t beta[N]) {
    int8_t fx[N];

    for (int Ni = 0; Ni < N; Ni++) {
        fx[Ni] = H[Ni] * beta[Ni];
    }

    return fx;
}

bool SampleCondPredictor::predict_using_given_hist(
    uint64_t seq_no,
    uint8_t piece,
    uint64_t PC,
    const SampleHist& hist_to_use,
    const bool pred_time_predict
) {
    int8_t x[N];
    int8_t a[H];

    build_features(PC, hist_to_use, x);

    int score = 0;

    for (int h = 0; h < H; ++h) {
        int sum = b[h];
        for (int i = 0; i < N; ++i) {
            sum += W[h][i] * x[i];
        }

        a[h] = (sum >= 0) ? 1 : -1;
        score += beta[h] * a[h];
    }

    // return score >= 0; // only use elm
    return (score >= 0) ? hist_to_use.tage_pred : !hist_to_use.tage_pred; // overwrite tage if predicted that it is wrong
}

void SampleCondPredictor::update(
    uint64_t PC,
    bool resolveDir,
    bool pred_taken,
    uint64_t nextPC,
    const SampleHist& hist_to_use
) {
    int8_t x[N];
    int8_t a[H];

    build_features(PC, hist_to_use, x);

    int score = 0;

    for (int h = 0; h < H; ++h) {
        int sum = b[h];
        for (int i = 0; i < N; ++i) {
            sum += W[h][i] * x[i];
        }

        a[h] = (sum >= 0) ? 1 : -1;
        score += beta[h] * a[h];
    }

    // int target = resolveDir ? 1 : -1; // traing for correct prediction
    int target = (hist_to_use.tage_pred == resolveDir) ? 1 : -1; // train to when tage is wrong

    if ((pred_taken != resolveDir) || (score >= -8 && score <= 8)) {
        for (int h = 0; h < H; ++h) {
            beta[h] += target * a[h];
        }
    }
}