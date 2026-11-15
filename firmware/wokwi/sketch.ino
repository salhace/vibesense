// VibeSense on ESP32 (Wokwi simulation).
// The "machine" is simulated in software: the potentiometer sets fault severity,
// the button cycles the machine condition. The ESP32 runs the same feature extraction
// and tiny forest model as the Python project and prints the result every second.
#include "vibesense_core.h"

const int PIN_POT = 34;
const int PIN_BTN = 15;
const int PIN_LED_G = 25;
const int PIN_LED_Y = 26;
const int PIN_LED_R = 27;

static float window[VS_WINDOW];
int condition = VS_HEALTHY;
unsigned long lastPress = 0;

void setup() {
  Serial.begin(115200);
  pinMode(PIN_BTN, INPUT_PULLUP);
  pinMode(PIN_LED_G, OUTPUT);
  pinMode(PIN_LED_Y, OUTPUT);
  pinMode(PIN_LED_R, OUTPUT);
  vs_seed(micros());
  Serial.println();
  Serial.println("VibeSense | simulated machine, tiny model on ESP32");
  Serial.println("Button: change condition | Potentiometer: fault severity");
}

void loop() {
  if (digitalRead(PIN_BTN) == LOW && millis() - lastPress > 300) {
    condition = (condition + 1) % N_CLASSES;
    lastPress = millis();
  }
  float severity = analogRead(PIN_POT) / 4095.0f;
  if (condition == VS_HEALTHY) severity = 0;

  vs_make_window(condition, severity, window);   // one second of simulated vibration

  unsigned long t0 = micros();
  vs_result r;
  vs_analyze(window, &r);                        // features + model
  unsigned long dt = micros() - t0;

  digitalWrite(PIN_LED_G, r.health >= 80);
  digitalWrite(PIN_LED_Y, r.health < 80 && r.health >= 55);
  digitalWrite(PIN_LED_R, r.health < 55);

  Serial.printf("true=%-13s sev=%.2f | model=%-13s conf=%3.0f%% health=%3.0f/100 | %lu us\n",
                VS_NAMES[condition], severity, VS_NAMES[r.state], r.prob[r.state] * 100.0f, r.health, dt);
  delay(1000);
}
