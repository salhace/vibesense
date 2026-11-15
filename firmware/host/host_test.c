// Desktop test harness for vibesense_core.h (compile with gcc, see tests/test_firmware.py).
//   host_test features   < windows.txt   -> prints features, class probabilities, severity per window
//   host_test selftest                   -> classifies windows made by the C simulator itself
#include <stdio.h>
#include <string.h>
#include "vibesense_core.h"

int main(int argc, char **argv) {
  static float w[VS_WINDOW];
  vs_result r;
  if (argc > 1 && !strcmp(argv[1], "features")) {
    for (;;) {
      for (int i = 0; i < VS_WINDOW; i++) if (scanf("%f", &w[i]) != 1) return 0;
      vs_analyze(w, &r);
      for (int i = 0; i < N_FEATURES; i++) printf("%.9g ", r.feat[i]);
      for (int c = 0; c < N_CLASSES; c++) printf("%.9g ", r.prob[c]);
      printf("%.9g\n", r.severity);
    }
  }
  if (argc > 1 && !strcmp(argv[1], "selftest")) {
    vs_seed(12345);
    int ok = 0, total = 0, per_ok[4] = {0}, per_n[4] = {0};
    float sev_err = 0;
    for (int rep = 0; rep < 60; rep++)
      for (int s = 0; s < 4; s++) {
        float sev = s == 0 ? 0.0f : vs_uniform(0.3f, 1.0f);
        vs_make_window(s, sev, w);
        vs_analyze(w, &r);
        total++; per_n[s]++;
        if (r.state == s) { ok++; per_ok[s]++; }
        sev_err += fabsf(r.severity - sev);
      }
    printf("accuracy %.4f\n", (double)ok / total);
    for (int s = 0; s < 4; s++) printf("%s %d/%d\n", VS_NAMES[s], per_ok[s], per_n[s]);
    printf("severity_mae %.4f\n", sev_err / total);
    return 0;
  }
  fprintf(stderr, "usage: host_test features|selftest\n");
  return 2;
}
