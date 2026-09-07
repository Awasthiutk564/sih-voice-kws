#include <WiFi.h>
#include <WebSocketsClient.h>
#include <driver/i2s.h>

// WiFi Configuration
const char* WIFI_SSID = "SIH-2026";
const char* WIFI_PASSWORD = "$1h@2026";

// Server Configuration
const char* SERVER_IP = "10.2.42.78";
const uint16_t SERVER_PORT = 8766;
const char* SERVER_PATH = "/";

// I2S Configuration for INMP441 (confirmed working pins)
#define I2S_WS 5
#define I2S_SD 4
#define I2S_SCK 6
#define I2S_PORT I2S_NUM_0

#define VOLUME_THRESHOLD 5000

WebSocketsClient webSocket;
bool isStreaming = false;
unsigned long streamStartTime = 0;
const unsigned long STREAM_DURATION = 5000;

void setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 1024,
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

bool isWakeWordDetected(int16_t* samples, size_t num_samples) {
  long long sum = 0;
  for(size_t i = 0; i < num_samples; i++) {
    sum += abs(samples[i]);
  }
  long avg = sum / num_samples;

  if (avg > VOLUME_THRESHOLD) {
    Serial.print("Loud sound detected! Avg volume: ");
    Serial.println(avg);
    return true;
  }
  return false;
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("[WSc] Disconnected!");
      break;
    case WStype_CONNECTED:
      Serial.printf("[WSc] Connected to url: %s\n", payload);
      break;
    case WStype_TEXT:
    case WStype_BIN:
      break;
  }
}

void setup() {
  Serial.begin(115200);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  setupI2S();

  webSocket.begin(SERVER_IP, SERVER_PORT, SERVER_PATH);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
}

void loop() {
  webSocket.loop();

  int16_t sampleBuffer[512];
  size_t bytesIn = 0;
  esp_err_t result = i2s_read(I2S_PORT, &sampleBuffer, sizeof(sampleBuffer), &bytesIn, portMAX_DELAY);

  if (result == ESP_OK && bytesIn > 0) {
    if (!isStreaming) {
      size_t num_samples = bytesIn / sizeof(int16_t);
      if (isWakeWordDetected(sampleBuffer, num_samples)) {
        isStreaming = true;
        streamStartTime = millis();
        Serial.println("WAKE TRIGGERED! Starting stream...");

        String wakeMsg = "WAKE:" + String(millis());
        webSocket.sendTXT(wakeMsg);
      }
    }

    if (isStreaming) {
      webSocket.sendBIN((uint8_t*)sampleBuffer, bytesIn);

      if (millis() - streamStartTime > STREAM_DURATION) {
        isStreaming = false;
        Serial.println("Stopping stream. Waiting for wake word...");
      }
    }
  }
}