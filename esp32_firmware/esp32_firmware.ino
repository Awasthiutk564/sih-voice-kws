/*
 * ISRO SIH Voice Activator - ESP32-S3 Firmware
 * 
 * Wiring INMP441 to ESP32-S3:
 * VDD -> 3.3V
 * GND -> GND
 * L/R -> GND
 * WS  -> GPIO 15
 * SCK -> GPIO 16 (BCLK)
 * SD  -> GPIO 17 (DATA)
 */

#include <WiFi.h>
#include <WebSocketsClient.h> // install via Arduino Library Manager (by Markus Sattler)
#include <driver/i2s.h>

// WiFi Configuration
const char* WIFI_SSID = "Lucifer";
const char* WIFI_PASSWORD = "Lucifer@12345";

// Server Configuration
const char* SERVER_IP = "192.168.137.1"; // e.g. "192.168.1.100"
const uint16_t SERVER_PORT = 8766;
const char* SERVER_PATH = "/";

// I2S Configuration for INMP441
#define I2S_WS 15
#define I2S_SD 17
#define I2S_SCK 16
#define I2S_PORT I2S_NUM_0

// Threshold for placeholder WAKE trigger (adjust based on noise)
#define VOLUME_THRESHOLD 5000 

WebSocketsClient webSocket;
bool isStreaming = false;
unsigned long streamStartTime = 0;
const unsigned long STREAM_DURATION = 5000; // Stream for 5 seconds after wake

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

// TODO (Model Trainer): Replace this body with TFLite KWS logic
bool isWakeWordDetected(int16_t* samples, size_t num_samples) {
  // Placeholder logic: trigger if average volume is loud enough
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
  
  // Connect to WiFi
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

  // Read I2S data (16-bit PCM mono)
  int16_t sampleBuffer[512];
  size_t bytesIn = 0;
  esp_err_t result = i2s_read(I2S_PORT, &sampleBuffer, sizeof(sampleBuffer), &bytesIn, portMAX_DELAY);
  
  if (result == ESP_OK && bytesIn > 0) {
    if (!isStreaming) {
      // Check for Wake Word
      size_t num_samples = bytesIn / sizeof(int16_t);
      if (isWakeWordDetected(sampleBuffer, num_samples)) {
        isStreaming = true;
        streamStartTime = millis();
        Serial.println("WAKE TRIGGERED! Starting stream...");
        
        // Send WAKE message with timestamp
        String wakeMsg = "WAKE:" + String(millis());
        webSocket.sendTXT(wakeMsg);
      }
    } 
    
    if (isStreaming) {
      // Stream raw binary audio to server
      webSocket.sendBIN((uint8_t*)sampleBuffer, bytesIn);
      
      // Stop streaming after duration
      if (millis() - streamStartTime > STREAM_DURATION) {
        isStreaming = false;
        Serial.println("Stopping stream. Waiting for wake word...");
      }
    }
  }
}
