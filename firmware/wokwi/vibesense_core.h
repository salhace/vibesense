// VibeSense core: vibration simulator + feature extraction + tiny forest inference.
// Plain C/C++, no libraries besides <math.h>. Works on a PC (gcc) and on ESP32 (Arduino).
// Everything here is driven by SIMULATED vibration (see vibesense/simulator.py).
#ifndef VIBESENSE_CORE_H
#define VIBESENSE_CORE_H

#include <math.h>
#include <stdint.h>
#include "model.h"

#define VS_FS 1024
#define VS_WINDOW 1024
#define VS_SHAFT_HZ 50.0f
#define VS_PI2 6.28318530718f

enum { VS_HEALTHY = 0, VS_IMBALANCE = 1, VS_BEARING = 2, VS_LOOSENESS = 3 };
static const char *VS_NAMES[N_CLASSES] = {"healthy", "imbalance", "bearing_fault", "looseness"};

// ---------------------------------------------------------------- random numbers
static uint32_t vs_rng_state = 2463534242u;
static inline void vs_seed(uint32_t s) { vs_rng_state = s ? s : 2463534242u; }
static inline uint32_t vs_rand_u32(void) {
  uint32_t x = vs_rng_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  return vs_rng_state = x;
}
static inline float vs_uniform(float a, float b) { return a + (b - a) * ((vs_rand_u32() >> 8) * (1.0f / 16777216.0f)); }
static inline float vs_gauss(void) {
  float u1 = vs_uniform(1e-7f, 1.0f), u2 = vs_uniform(0.0f, 1.0f);
  return sqrtf(-2.0f * logf(u1)) * cosf(VS_PI2 * u2);
}

// ---------------------------------------------------------------- simulator
// state: 0..3, sev: 0..1 (ignored for healthy). Fills x[VS_WINDOW] with acceleration in g.
static inline void vs_make_window(int state, float sev, float *x) {
  if (sev < 0) sev = 0;
  if (sev > 1) sev = 1;
  float shaft = VS_SHAFT_HZ + 0.8f * vs_gauss();
  float phase = vs_uniform(0, VS_PI2);
  for (int i = 0; i < VS_WINDOW; i++) {
    float t = (float)i / VS_FS;
    x[i] = 0.25f * sinf(VS_PI2 * shaft * t + phase) + 0.05f * vs_gauss();
  }
  if (state == VS_IMBALANCE) {
    float a = 0.3f + 1.6f * sev;
    for (int i = 0; i < VS_WINDOW; i++) x[i] += a * sinf(VS_PI2 * shaft * ((float)i / VS_FS) + phase);
  } else if (state == VS_BEARING) {
    float impact_hz = 87.0f + vs_gauss();
    float amp = 0.15f + 1.4f * sev;
    int n_imp = (int)impact_hz;
    for (int k = 0; k < n_imp; k++) {
      float t0 = (float)k / impact_hz + 0.0005f * vs_gauss();
      int idx = (int)(t0 * VS_FS);
      if (idx < 0) idx = 0;
      if (idx >= VS_WINDOW - 1) continue;
      for (int j = 0; j < 64 && idx + j < VS_WINDOW; j++) {  // the ring-down is gone after ~64 samples
        float tail = (float)j / VS_FS;
        x[idx + j] += amp * expf(-tail * 700.0f) * sinf(VS_PI2 * 300.0f * tail);
      }
    }
    float sg = 0.03f + 0.1f * sev;
    for (int i = 0; i < VS_WINDOW; i++) x[i] += sg * vs_gauss();
  } else if (state == VS_LOOSENESS) {
    static const float H[4] = {2.0f, 3.0f, 4.0f, 0.5f};
    static const float A[4] = {0.7f, 0.5f, 0.35f, 0.3f};
    for (int h = 0; h < 4; h++) {
      float a = A[h] * (0.2f + 1.2f * sev), ph = vs_uniform(0, VS_PI2);
      for (int i = 0; i < VS_WINDOW; i++) x[i] += a * sinf(VS_PI2 * shaft * H[h] * ((float)i / VS_FS) + ph);
    }
  }
  // sensor/mount variation, same as the training data
  float gain = vs_uniform(0.8f, 1.25f), ns = vs_uniform(0.0f, 0.12f);
  for (int i = 0; i < VS_WINDOW; i++) x[i] = x[i] * gain + ns * vs_gauss();
}

// ---------------------------------------------------------------- features
static const int VS_BANDS[6][2] = {{0, 30}, {30, 70}, {70, 130}, {130, 230}, {230, 380}, {380, 500}};

static inline void vs_fft(float *re, float *im, int n) {
  for (int i = 1, j = 0; i < n; i++) {  // bit reversal
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (int len = 2; len <= n; len <<= 1) {
    float ang = -VS_PI2 / len;
    for (int i = 0; i < n; i += len) {
      for (int k = 0; k < len / 2; k++) {
        float wr = cosf(ang * k), wi = sinf(ang * k);
        int a = i + k, b = i + k + len / 2;
        float xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

// f[N_FEATURES] = rms, peak, crest, kurtosis, 6 band energies, peak frequency (same order as features.py)
static inline void vs_features(const float *x, float *f) {
  static float re[VS_WINDOW], im[VS_WINDOW];
  const int N = VS_WINDOW;
  double sum = 0, sq = 0;
  float peak = 0;
  for (int i = 0; i < N; i++) { sum += x[i]; sq += (double)x[i] * x[i]; float a = fabsf(x[i]); if (a > peak) peak = a; }
  float mean = (float)(sum / N), rms = sqrtf((float)(sq / N));
  double m2 = 0, m4 = 0;
  for (int i = 0; i < N; i++) { double c = x[i] - mean; m2 += c * c; m4 += c * c * c * c; }
  m2 /= N; m4 /= N;
  f[0] = rms; f[1] = peak; f[2] = peak / (rms + 1e-9f); f[3] = (float)(m4 / (m2 * m2 + 1e-12));
  double wsum = 0;
  for (int i = 0; i < N; i++) {
    float w = 0.5f - 0.5f * cosf(VS_PI2 * i / (N - 1));
    re[i] = (x[i] - mean) * w; im[i] = 0; wsum += w;
  }
  vs_fft(re, im, N);
  float scale = 2.0f / (float)wsum;
  for (int b = 0; b < 6; b++) {
    double e = 0;
    for (int k = VS_BANDS[b][0]; k < VS_BANDS[b][1]; k++) {
      float m = sqrtf(re[k] * re[k] + im[k] * im[k]) * scale;
      e += (double)m * m;
    }
    f[4 + b] = sqrtf((float)e);
  }
  float best = -1; int bk = 1;
  for (int k = 1; k <= N / 2; k++) {
    float m = re[k] * re[k] + im[k] * im[k];
    if (m > best) { best = m; bk = k; }
  }
  f[10] = (float)bk;  // 1 Hz per bin
}

// ---------------------------------------------------------------- tiny forest
static inline int vs_walk_clf(int n, const float *f) {
  while (CLF_FEAT[n] >= 0) n = (f[CLF_FEAT[n]] <= CLF_THR[n]) ? CLF_LEFT[n] : CLF_RIGHT[n];
  return n;
}
static inline int vs_walk_reg(int n, const float *f) {
  while (REG_FEAT[n] >= 0) n = (f[REG_FEAT[n]] <= REG_THR[n]) ? REG_LEFT[n] : REG_RIGHT[n];
  return n;
}

typedef struct {
  int state;                 // index into VS_NAMES
  float prob[N_CLASSES];     // class probabilities
  float severity;            // 0..1
  float health;              // 0..100
  float feat[N_FEATURES];
} vs_result;

static inline void vs_analyze(const float *x, vs_result *r) {
  vs_features(x, r->feat);
  for (int c = 0; c < N_CLASSES; c++) r->prob[c] = 0;
  for (int t = 0; t < CLF_TREES; t++) {
    int leaf = vs_walk_clf(CLF_ROOT[t], r->feat);
    for (int c = 0; c < N_CLASSES; c++) r->prob[c] += CLF_VAL[leaf][c] / CLF_TREES;
  }
  int best = 0;
  for (int c = 1; c < N_CLASSES; c++) if (r->prob[c] > r->prob[best]) best = c;
  r->state = best;
  float s = 0;
  for (int t = 0; t < REG_TREES; t++) s += REG_VAL[vs_walk_reg(REG_ROOT[t], r->feat)] / REG_TREES;
  if (s < 0) s = 0;
  if (s > 1) s = 1;
  r->severity = s;
  r->health = 100.0f * (1.0f - s);
}

#endif
