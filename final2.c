#include <WiFi.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <TinyGPS++.h>
#include <ESP_Mail_Client.h>
#include "model_params.h"
#include <HTTPClient.h>
#include <ArduinoJson.h>

// Wi-Fi and Email Configuration
#define WIFI_SSID "YourWiFiName"
#define WIFI_PASSWORD "YourWiFiPassword"
#define SMTP_HOST "smtp.gmail.com"
#define SMTP_PORT 465
#define AUTHOR_EMAIL "svijayaragavan185@gmail.com"
#define AUTHOR_PASSWORD "your-app-password" // Use app password for Gmail
#define RECIPIENT_EMAIL "vs6550@srmist.edu.in"

// Create sensor objects
Adafruit_MPU6050 mpu1; // Front sensor
Adafruit_MPU6050 mpu2; // Rear sensor
TinyGPSPlus gps;

// Setup email session
SMTPSession smtp;

// Define pins
#define MPU1_SDA 21
#define MPU1_SCL 22
#define MPU2_SDA 25 // Using different pins for second I2C bus
#define MPU2_SCL 26
#define GPS_RX 16
#define GPS_TX 17
#define LED_PIN 2 // Built-in LED

// Set up two I2C ports
TwoWire I2C_ONE = TwoWire(0);
TwoWire I2C_TWO = TwoWire(1);

// Buffer for sensor readings
const int WINDOW_SIZE = 50; // Match training window size
struct SensorData {
  float acc_x1, acc_y1, acc_z1; // Front sensor
  float acc_x2, acc_y2, acc_z2; // Rear sensor
  float gyr_x1, gyr_y1, gyr_z1; // Front sensor
  float gyr_x2, gyr_y2, gyr_z2; // Rear sensor
  float acc_mag1, acc_mag2; // Magnitudes
};

SensorData buffer[WINDOW_SIZE];
int buffer_index = 0;
bool buffer_filled = false;

// GPS variables
float current_lat = 0.0;
float current_lng = 0.0;
bool gps_fixed = false;

// Features array
float features[NUM_FEATURES];

// Pothole detection parameters
const int MIN_POTHOLE_INTERVAL = 5000; // 5 seconds between detections
unsigned long last_pothole_time = 0;
int pothole_count = 0;

// Email parameters
const int EMAIL_LIMIT = 10; // Max emails per session
int email_count = 0;
bool email_enabled = false;

// Exponential moving average coefficients for filtering
const float alpha = 0.1; // Lower = more smoothing
float ema_acc_x1 = 0, ema_acc_y1 = 0, ema_acc_z1 = 0; // Front sensor
float ema_acc_x2 = 0, ema_acc_y2 = 0, ema_acc_z2 = 0; // Rear sensor

// Callback function for email sending status
void smtpCallback(SMTP_Status status) {
  Serial.println(status.info());
  if (status.success()) {
    Serial.println("Email sent successfully");
  } else {
    Serial.println("Error sending email");
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("Pothole Detection System Starting");

  // Initialize LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Initialize GPS
  Serial2.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);
  Serial.println("GPS initialized");

  // Initialize first MPU6050
  I2C_ONE.begin(MPU1_SDA, MPU1_SCL);
  if (!mpu1.begin(0x68, &I2C_ONE)) {
    Serial.println("Failed to find first MPU6050 chip");
    blinkLED(2); // Error code
    while (1) {
      delay(10);
    }
  } else {
    Serial.println("MPU6050 #1 (Front) Found!");
    mpu1.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu1.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu1.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }

  // Initialize second MPU6050
  I2C_TWO.begin(MPU2_SDA, MPU2_SCL);
  if (!mpu2.begin(0x68, &I2C_TWO)) {
    Serial.println("Failed to find second MPU6050 chip");
    blinkLED(3); // Error code
    while (1) {
      delay(10);
    }
  } else {
    Serial.println("MPU6050 #2 (Rear) Found!");
    mpu2.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu2.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu2.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }

  // Connect to WiFi
  Serial.print("Connecting to WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  // Wait for connection with timeout
  int wifi_attempt = 0;
  while (WiFi.status() != WL_CONNECTED && wifi_attempt < 20) {
    delay(500);
    Serial.print(".");
    wifi_attempt++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    email_enabled = true;
    blinkLED(5); // Success code for WiFi
  } else {
    Serial.println("\nWiFi connection failed, continuing without email");
  }

  // Initialize email settings
  if (email_enabled) {
    smtp.debug(1);
    smtp.callback(smtpCallback);
  }

  // System ready
  blinkLED(1);
  Serial.println("System ready!");
}

void loop() {
  // Update GPS data
  while (Serial2.available() > 0) {
    if (gps.encode(Serial2.read())) {
      if (gps.location.isValid()) {
        current_lat = gps.location.lat();
        current_lng = gps.location.lng();
        gps_fixed = true;
      }
    }
  }

  // Read from sensors
  sensors_event_t a1, g1, temp1; // Front sensor
  sensors_event_t a2, g2, temp2; // Rear sensor
  mpu1.getEvent(&a1, &g1, &temp1);
  mpu2.getEvent(&a2, &g2, &temp2);

  // Apply exponential moving average filter for noise reduction
  ema_acc_x1 = alpha * a1.acceleration.x + (1 - alpha) * ema_acc_x1;
  ema_acc_y1 = alpha * a1.acceleration.y + (1 - alpha) * ema_acc_y1;
  ema_acc_z1 = alpha * a1.acceleration.z + (1 - alpha) * ema_acc_z1;
  ema_acc_x2 = alpha * a2.acceleration.x + (1 - alpha) * ema_acc_x2;
  ema_acc_y2 = alpha * a2.acceleration.y + (1 - alpha) * ema_acc_y2;
  ema_acc_z2 = alpha * a2.acceleration.z + (1 - alpha) * ema_acc_z2;

  // Update buffer with both sensor readings
  buffer[buffer_index].acc_x1 = ema_acc_x1;
  buffer[buffer_index].acc_y1 = ema_acc_y1;
  buffer[buffer_index].acc_z1 = ema_acc_z1;
  buffer[buffer_index].gyr_x1 = g1.gyro.x;
  buffer[buffer_index].gyr_y1 = g1.gyro.y;
  buffer[buffer_index].gyr_z1 = g1.gyro.z;
  buffer[buffer_index].acc_x2 = ema_acc_x2;
  buffer[buffer_index].acc_y2 = ema_acc_y2;
  buffer[buffer_index].acc_z2 = ema_acc_z2;
  buffer[buffer_index].gyr_x2 = g2.gyro.x;
  buffer[buffer_index].gyr_y2 = g2.gyro.y;
  buffer[buffer_index].gyr_z2 = g2.gyro.z;

  // Calculate magnitudes for both sensors
  buffer[buffer_index].acc_mag1 = sqrt(ema_acc_x1*ema_acc_x1 +
                                     ema_acc_y1*ema_acc_y1 +
                                     ema_acc_z1*ema_acc_z1);
  buffer[buffer_index].acc_mag2 = sqrt(ema_acc_x2*ema_acc_x2 +
                                     ema_acc_y2*ema_acc_y2 +
                                     ema_acc_z2*ema_acc_z2);

  // Update buffer index
  buffer_index = (buffer_index + 1) % WINDOW_SIZE;
  if (buffer_index == 0) {
    buffer_filled = true;
  }

  // Check for pothole if buffer is filled
  if (buffer_filled && (millis() - last_pothole_time > MIN_POTHOLE_INTERVAL)) {
    extract_features();
    bool is_pothole = detect_pothole();
    if (is_pothole) {
      pothole_detected();
      last_pothole_time = millis();
    }
  }

  // Status update every second
  static unsigned long last_status = 0;
  if (millis() - last_status >= 1000) {
    Serial.print("Front: ");
    Serial.print(a1.acceleration.z);
    Serial.print(" | Rear: ");
    Serial.print(a2.acceleration.z);
    Serial.print(" | Mag1: ");
    Serial.print(buffer[buffer_index].acc_mag1);
    Serial.print(" | Mag2: ");
    Serial.print(buffer[buffer_index].acc_mag2);
    Serial.print(" | Potholes: ");
    Serial.print(pothole_count);
    if (gps_fixed) {
      Serial.print(" | GPS: ");
      Serial.print(current_lat, 6);
      Serial.print(", ");
      Serial.print(current_lng, 6);
    } else {
      Serial.print(" | GPS: Waiting for fix");
    }

    if (email_enabled) {
      Serial.print(" | Email: Enabled");
    } else {
      Serial.print(" | Email: Disabled");
    }

    Serial.println();
    last_status = millis();
  }

  // Reconnect WiFi if needed
  if (WiFi.status() != WL_CONNECTED && email_enabled) {
    Serial.println("WiFi disconnected. Attempting to reconnect...");
    WiFi.reconnect();
    int reconnect_attempt = 0;
    while (WiFi.status() != WL_CONNECTED && reconnect_attempt < 10) {
      delay(500);
      Serial.print(".");
      reconnect_attempt++;
    }

    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\nWiFi reconnected");
    } else {
      Serial.println("\nWiFi reconnection failed, disabling email");
      email_enabled = false;
    }
  }

  // Delay to achieve approximately 100Hz sampling
  delay(10);
}

void send_pothole_to_server() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Cannot send data - WiFi not connected");
    return;
  }

  Serial.println("Sending pothole data to server...");
  // Create JSON payload
  StaticJsonDocument<1024> doc;
  JsonArray accelerometer_data = doc.createNestedArray("accelerometer_data");
  
  // Add last 50 samples to the payload
  for (int i = 0; i < WINDOW_SIZE; i++) {
    JsonObject sample = accelerometer_data.createNestedObject();
    int idx = (buffer_index + i) % WINDOW_SIZE;
    sample["acc_x1"] = buffer[idx].acc_x1;
    sample["acc_y1"] = buffer[idx].acc_y1;
    sample["acc_z1"] = buffer[idx].acc_z1;
    sample["acc_x2"] = buffer[idx].acc_x2;
    sample["acc_y2"] = buffer[idx].acc_y2;
    sample["acc_z2"] = buffer[idx].acc_z2;
    sample["gyr_x1"] = buffer[idx].gyr_x1;
    sample["gyr_y1"] = buffer[idx].gyr_y1;
    sample["gyr_z1"] = buffer[idx].gyr_z1;
    sample["gyr_x2"] = buffer[idx].gyr_x2;
    sample["gyr_y2"] = buffer[idx].gyr_y2;
    sample["gyr_z2"] = buffer[idx].gyr_z2;
  }

  // Add GPS coordinates
  doc["latitude"] = current_lat;
  doc["longitude"] = current_lng;
  
  // Serialize JSON
  String json;
  serializeJson(doc, json);
  
  // Send HTTP POST request
  HTTPClient http;
  http.begin("http://your-server-ip:5000/api/detect");
  http.addHeader("Content-Type", "application/json");
  
  int httpResponseCode = http.POST(json);
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.println("HTTP Response code: " + String(httpResponseCode));
    Serial.println("Response: " + response);
  } else {
    Serial.println("Error on sending POST: " + String(httpResponseCode));
  }
  
  http.end();
}

void extract_features() {
  // Feature index
  int feat_idx = 0;
  
  // Time domain features for front sensor (acc_x1, acc_y1, acc_z1)
  for (const char* axis : {"acc_x1", "acc_y1", "acc_z1"}) {
    float min_val = 1000.0;
    float max_val = -1000.0;
    float sum = 0.0;
    float sum_sq = 0.0;
    
    // For each axis, select the right data from buffer
    for (int i = 0; i < WINDOW_SIZE; i++) {
      float val;
      if (strcmp(axis, "acc_x1") == 0) val = buffer[i].acc_x1;
      else if (strcmp(axis, "acc_y1") == 0) val = buffer[i].acc_y1;
      else val = buffer[i].acc_z1;
      
      min_val = min(min_val, val);
      max_val = max(max_val, val);
      sum += val;
      sum_sq += val * val;
    }
    
    float mean = sum / WINDOW_SIZE;
    float variance = (sum_sq / WINDOW_SIZE) - (mean * mean);
    float std_dev = sqrt(variance);
    
    // Add to features array
    features[feat_idx++] = min_val; // min
    features[feat_idx++] = max_val; // max
    features[feat_idx++] = mean; // mean
    features[feat_idx++] = std_dev; // std
  }
  
  // Time domain features for rear sensor (acc_x2, acc_y2, acc_z2)
  for (const char* axis : {"acc_x2", "acc_y2", "acc_z2"}) {
    float min_val = 1000.0;
    float max_val = -1000.0;
    float sum = 0.0;
    float sum_sq = 0.0;
    
    // For each axis, select the right data from buffer
    for (int i = 0; i < WINDOW_SIZE; i++) {
      float val;
      if (strcmp(axis, "acc_x2") == 0) val = buffer[i].acc_x2;
      else if (strcmp(axis, "acc_y2") == 0) val = buffer[i].acc_y2;
      else val = buffer[i].acc_z2;
      
      min_val = min(min_val, val);
      max_val = max(max_val, val);
      sum += val;
      sum_sq += val * val;
    }
    
    float mean = sum / WINDOW_SIZE;
    float variance = (sum_sq / WINDOW_SIZE) - (mean * mean);
    float std_dev = sqrt(variance);
    
    // Add to features array
    features[feat_idx++] = min_val; // min
    features[feat_idx++] = max_val; // max
    features[feat_idx++] = mean; // mean
    features[feat_idx++] = std_dev; // std
  }
  
  // Magnitude features
  // Front sensor magnitude
  float mag1_max = -1000.0;
  float mag1_sum = 0.0;
  float mag1_sum_sq = 0.0;
  
  // Rear sensor magnitude
  float mag2_max = -1000.0;
  float mag2_sum = 0.0;
  float mag2_sum_sq = 0.0;
  
  for (int i = 0; i < WINDOW_SIZE; i++) {
    // Front sensor
    mag1_max = max(mag1_max, buffer[i].acc_mag1);
    mag1_sum += buffer[i].acc_mag1;
    mag1_sum_sq += buffer[i].acc_mag1 * buffer[i].acc_mag1;
    
    // Rear sensor
    mag2_max = max(mag2_max, buffer[i].acc_mag2);
    mag2_sum += buffer[i].acc_mag2;
    mag2_sum_sq += buffer[i].acc_mag2 * buffer[i].acc_mag2;
  }
  
  float mag1_mean = mag1_sum / WINDOW_SIZE;
  float mag1_std = sqrt((mag1_sum_sq / WINDOW_SIZE) - (mag1_mean * mag1_mean));
  float mag2_mean = mag2_sum / WINDOW_SIZE;
  float mag2_std = sqrt((mag2_sum_sq / WINDOW_SIZE) - (mag2_mean * mag2_mean));
  
  // Add magnitude features
  features[feat_idx++] = mag1_max;
  features[feat_idx++] = mag2_max;
  features[feat_idx++] = mag1_std;
  features[feat_idx++] = mag2_std;
  
  // Gyroscope features - Front sensor
  for (const char* axis : {"gyr_x1", "gyr_y1", "gyr_z1"}) {
    float sum = 0.0;
    float sum_sq = 0.0;
    
    for (int i = 0; i < WINDOW_SIZE; i++) {
      float val;
      if (strcmp(axis, "gyr_x1") == 0) val = buffer[i].gyr_x1;
      else if (strcmp(axis, "gyr_y1") == 0) val = buffer[i].gyr_y1;
      else val = buffer[i].gyr_z1;
      
      sum += val;
      sum_sq += val * val;
    }
    
    float mean = sum / WINDOW_SIZE;
    float variance = (sum_sq / WINDOW_SIZE) - (mean * mean);
    float std_dev = sqrt(variance);
    
    features[feat_idx++] = std_dev; // std
  }
  
  // Gyroscope features - Rear sensor
  for (const char* axis : {"gyr_x2", "gyr_y2", "gyr_z2"}) {
    float sum = 0.0;
    float sum_sq = 0.0;
    
    for (int i = 0; i < WINDOW_SIZE; i++) {
      float val;
      if (strcmp(axis, "gyr_x2") == 0) val = buffer[i].gyr_x2;
      else if (strcmp(axis, "gyr_y2") == 0) val = buffer[i].gyr_y2;
      else val = buffer[i].gyr_z2;
      
      sum += val;
      sum_sq += val * val;
    }
    
    float mean = sum / WINDOW_SIZE;
    float variance = (sum_sq / WINDOW_SIZE) - (mean * mean);
    float std_dev = sqrt(variance);
    
    features[feat_idx++] = std_dev; // std
  }
}

bool detect_pothole() {
  // Apply SVM model
  float result = SVM_BIAS;
  for (int i = 0; i < NUM_FEATURES; i++) {
    result += SVM_WEIGHTS[i] * features[i];
  }
  
  // SVM decision
  return result > SVM_THRESHOLD;
}

void pothole_detected() {
  // Increment counter
  pothole_count++;
  
  // Flash LED
  digitalWrite(LED_PIN, HIGH);
  
  // Log detection
  Serial.println("==== POTHOLE DETECTED! ====");
  Serial.print("Total count: ");
  Serial.println(pothole_count);
  
  if (gps_fixed) {
    Serial.print("Location: ");
    Serial.print(current_lat, 6);
    Serial.print(", ");
    Serial.println(current_lng, 6);
    Serial.print("Google Maps Link: https://maps.google.com/?q=");
    Serial.print(current_lat, 6);
    Serial.print(",");
    Serial.println(current_lng, 6);
    
    // Send email notification if enabled and under limit
    if (email_enabled && email_count < EMAIL_LIMIT) {
      send_email_notification();
      email_count++;
    }
  }
  
  send_pothole_to_server();
  
  // Turn off LED after 500ms
  delay(500);
  digitalWrite(LED_PIN, LOW);
}

void send_email_notification() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Cannot send email - WiFi not connected");
    return;
  }

  Serial.println("Sending email notification...");
  ESP_Mail_Session session;
  session.server.host_name = SMTP_HOST;
  session.server.port = SMTP_PORT;
  session.login.email = AUTHOR_EMAIL;
  session.login.password = AUTHOR_PASSWORD;
  session.login.user_domain = "";

  SMTP_Message message;
  message.sender.name = "Pothole Detection System";
  message.sender.email = AUTHOR_EMAIL;
  message.subject = "Pothole Alert";
  message.addRecipient("Maintenance", RECIPIENT_EMAIL);

  String content = "Pothole Detection Report\n\n";
  content += "Location: " + String(current_lat, 6) + ", " + String(current_lng, 6) + "\n";
  content += "Google Maps Link: https://maps.google.com/?q=" + String(current_lat, 6) + "," + String(current_lng, 6) + "\n";
  content += "Time: " + String(millis() / 1000) + " seconds since start\n";
  content += "Pothole count: " + String(pothole_count) + "\n\n";
  content += "This is an automated message from your Pothole Detection System.";

  message.text.content = content.c_str();
  message.text.charSet = "us-ascii";
  message.text.transfer_encoding = Content_Transfer_Encoding::enc_7bit;

  if (!smtp.connect(&session))
    return;
  
  if (!MailClient.sendMail(&smtp, &message))
    Serial.println("Error sending email: " + smtp.errorReason());
}

void blinkLED(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    if (i < times - 1) {
      delay(200);
    }
  }
}
