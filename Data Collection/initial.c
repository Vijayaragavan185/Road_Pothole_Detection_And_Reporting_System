#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// Create sensor objects
Adafruit_MPU6050 mpu1; // Front sensor
Adafruit_MPU6050 mpu2; // Rear sensor

// Define pins
#define MPU1_SDA 21
#define MPU1_SCL 22
#define MPU2_SDA 25  // Using different pins for second I2C bus
#define MPU2_SCL 26

// Set up two I2C ports
TwoWire I2C_ONE = TwoWire(0);
TwoWire I2C_TWO = TwoWire(1);

// Data collection variables
unsigned long start_time;
int sample_count = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("Pothole Data Collection System");

  // Initialize first MPU6050
  I2C_ONE.begin(MPU1_SDA, MPU1_SCL);
  if (!mpu1.begin(0x68, &I2C_ONE)) {
    Serial.println("Failed to find first MPU6050 chip");
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
    while (1) {
      delay(10);
    }
  } else {
    Serial.println("MPU6050 #2 (Rear) Found!");
    mpu2.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu2.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu2.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }
  
  // Print CSV header
  Serial.println("timestamp,acc_x1,acc_y1,acc_z1,acc_x2,acc_y2,acc_z2,gyr_x1,gyr_y1,gyr_z1,gyr_x2,gyr_y2,gyr_z2,is_pothole");
  
  start_time = millis();
}

void loop() {
  // Read sensor data
  sensors_event_t a1, g1, temp1;  // Front sensor
  sensors_event_t a2, g2, temp2;  // Rear sensor
  
  mpu1.getEvent(&a1, &g1, &temp1);
  mpu2.getEvent(&a2, &g2, &temp2);

  // Log data (100Hz sampling rate)
  static unsigned long last_sample_time = 0;
  if (millis() - last_sample_time >= 10) { // 10ms = 100Hz
    // Format as JSON-like structure for easier parsing
    Serial.print("{\"ax1\":");
    Serial.print(a1.acceleration.x);
    Serial.print(",\"ay1\":");
    Serial.print(a1.acceleration.y);
    Serial.print(",\"az1\":");
    Serial.print(a1.acceleration.z);
    Serial.print(",\"gx1\":");
    Serial.print(g1.gyro.x);
    Serial.print(",\"gy1\":");
    Serial.print(g1.gyro.y);
    Serial.print(",\"gz1\":");
    Serial.print(g1.gyro.z);
    
    Serial.print(",\"ax2\":");
    Serial.print(a2.acceleration.x);
    Serial.print(",\"ay2\":");
    Serial.print(a2.acceleration.y);
    Serial.print(",\"az2\":");
    Serial.print(a2.acceleration.z);
    Serial.print(",\"gx2\":");
    Serial.print(g2.gyro.x);
    Serial.print(",\"gy2\":");
    Serial.print(g2.gyro.y);
    Serial.print(",\"gz2\":");
    Serial.print(g2.gyro.z);
    
    Serial.print(",\"t\":");
    Serial.print(millis() - start_time);
    Serial.println("}");
    
    sample_count++;
    last_sample_time = millis();
  }
  
  // Small delay for stability
  delay(1);
}