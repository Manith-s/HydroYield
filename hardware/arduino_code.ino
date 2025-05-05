/**
 * Smart Hydroponic System - Sensor Data Collection
 * This Arduino code collects data from various sensors and sends it to the ESP8266 WiFi module
 */

// Include required libraries
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <SoftwareSerial.h>

// Define pins for sensors
#define TEMP_SENSOR_PIN 2      // DS18B20 temperature sensor
#define PH_SENSOR_PIN A0       // pH sensor analog pin
#define HUMIDITY_SENSOR_PIN A1 // Humidity sensor analog pin
#define WATER_LEVEL_PIN A2     // Water level sensor

// Define pins for communication with ESP8266
#define ESP_RX 10
#define ESP_TX 11x

// Setup OneWire instance for temperature sensor
OneWire oneWire(TEMP_SENSOR_PIN);
DallasTemperature sensors(&oneWire);

// Software serial for ESP8266 communication
SoftwareSerial espSerial(ESP_RX, ESP_TX);

// Sensor reading variables
float temperature = 0.0;
float humidity = 0.0;
float ph = 0.0;
float waterLevel = 0.0;

// Timing variables
unsigned long previousMillis = 0;
const long interval = 10000; // Send data every 10 seconds

void setup() {
  // Initialize serial communications
  Serial.begin(9600);
  espSerial.begin(9600);
  
  // Initialize sensors
  sensors.begin();
  
  // Initialize analog pins
  pinMode(PH_SENSOR_PIN, INPUT);
  pinMode(HUMIDITY_SENSOR_PIN, INPUT);
  pinMode(WATER_LEVEL_PIN, INPUT);
  
  Serial.println("Smart Hydroponic System initialized");
}

void loop() {
  unsigned long currentMillis = millis();
  
  // Check if it's time to read sensors and send data
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;
    
    // Read sensors
    readSensors();
    
    // Send data to ESP8266
    sendDataToESP();
    
    // Debug output to serial monitor
    printSensorData();
  }
}

void readSensors() {
  // Read temperature
  sensors.requestTemperatures();
  temperature = sensors.getTempCByIndex(0);
  
  // Read pH (conversion from analog to pH)
  int phValue = analogRead(PH_SENSOR_PIN);
  ph = convertToPH(phValue);
  
  // Read humidity (conversion from analog to percentage)
  int humidityValue = analogRead(HUMIDITY_SENSOR_PIN);
  humidity = convertToHumidity(humidityValue);
  
  // Read water level
  int waterLevelValue = analogRead(WATER_LEVEL_PIN);
  waterLevel = map(waterLevelValue, 0, 1023, 0, 100);
}

float convertToPH(int analogValue) {
  // Convert analog reading to pH (calibration needed)
  // This is a simplified example, actual conversion depends on sensor calibration
  float voltage = analogValue * (5.0 / 1024.0);
  return 3.5 * voltage;
}

float convertToHumidity(int analogValue) {
  // Convert analog reading to humidity percentage (calibration needed)
  // This is a simplified example, actual conversion depends on sensor calibration
  return map(analogValue, 0, 1023, 0, 100);
}

void sendDataToESP() {
  // Format data as JSON
  String data = "{\"temperature\":" + String(temperature) + 
                ",\"humidity\":" + String(humidity) + 
                ",\"ph\":" + String(ph) + 
                ",\"waterLevel\":" + String(waterLevel) + "}";
  
  // Send the data to ESP8266
  espSerial.println(data);
}

void printSensorData() {
  Serial.println("Sensor Readings:");
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");
  
  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.println(" %");
  
  Serial.print("pH: ");
  Serial.println(ph);
  
  Serial.print("Water Level: ");
  Serial.print(waterLevel);
  Serial.println(" %");
  
  Serial.println("---------------------------");
}