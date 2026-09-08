/*
 * ISRO SIH Voice Activator - High-Clarity Streaming ASR for ESP32-S3
 * 
 * ZERO EXTERNAL LIBRARIES REQUIRED!
 * High-fidelity 32-bit -> 16-bit conversion for INMP441 microphone.
 *
 * Microcontroller: ESP32-S3
 * Microphone: INMP441 I2S Digital Microphone
 *
 * Wiring INMP441 to ESP32-S3:
 * VDD -> 3.3V
 * GND -> GND
 * L/R -> GND (Left Channel)
 * WS  -> GPIO 5
 * SD  -> GPIO 4 (DATA)
 * SCK -> GPIO 6 (BCLK)
 */

#include <WiFi.h>
#include <driver/i2s.h>

// WiFi Configuration
const char* WIFI_SSID = "SIH-2026";
const char* WIFI_PASSWORD = "$1h@2026";

// Server Configuration (Laptop IP running server.py)
const char* SERVER_IP = "10.2.42.36";
const uint16_t SERVER_PORT = 8766;

// Confirmed working I2S pins for INMP441
#define I2S_WS   5
#define I2S_SD   4
#define I2S_SCK  6
#define I2S_PORT I2S_NUM_0

WiFiClient client;

void setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, // INMP441 uses 24-bit in 32-bit slot
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 512,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
}

void setup() {
  Serial.begin(115200);

  // Connect to Wi-Fi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n[+] WiFi Connected! ESP32 IP: " + WiFi.localIP().toString());

  // Setup I2S with 32-bit alignment
  setupI2S();
  Serial.println("[+] I2S High-Fidelity Audio Driver Initialized.");
}

void loop() {
  // Ensure connection to Python ASR Server
  if (!client.connected()) {
    Serial.print("[*] Connecting to ASR Server at ");
    Serial.print(SERVER_IP);
    Serial.print(":");
    Serial.println(SERVER_PORT);
    
    if (client.connect(SERVER_IP, SERVER_PORT)) {
      client.setNoDelay(true); // Disable TCP delay for instant streaming
      Serial.println("[+] Connected to Server! High-clarity audio streaming active...");
    } else {
      Serial.println("[-] Connection failed. Retrying in 2 seconds...");
      delay(2000);
      return;
    }
  }

  // Read 32-bit raw I2S samples from INMP441 with portMAX_DELAY
  int32_t rawBuffer[256];
  int16_t outBuffer[256];
  size_t bytesIn = 0;
  esp_err_t result = i2s_read(I2S_PORT, rawBuffer, sizeof(rawBuffer), &bytesIn, portMAX_DELAY);

  if (result == ESP_OK && bytesIn > 0 && client.connected()) {
    size_t samplesRead = bytesIn / sizeof(int32_t);
    for (size_t i = 0; i < samplesRead; i++) {
      // Natural 24-bit to 16-bit PCM shift (preserves audio dynamic range without clipping)
      int32_t sample = rawBuffer[i] >> 14;
      if (sample > 32767) sample = 32767;
      if (sample < -32768) sample = -32768;
      outBuffer[i] = (int16_t)sample;
    }
    // Stream clean 16-bit 16kHz PCM audio to server
    client.write((const uint8_t*)outBuffer, samplesRead * sizeof(int16_t));
  }
}
