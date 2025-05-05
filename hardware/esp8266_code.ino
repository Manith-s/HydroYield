/**
 * Smart Hydroponic System - ESP8266 WiFi Module
 * This code receives sensor data from Arduino and sends it to the cloud server
 */

 #include <ESP8266WiFi.h>
 #include <ESP8266HTTPClient.h>
 #include <ArduinoJson.h>
 #include <SoftwareSerial.h>
 
 // WiFi credentials
 #define WIFI_SSID "YOUR_WIFI_SSID"
 #define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
 
 // Server endpoint
 #define SERVER_URL "http://your-server-url.com/api/sensor-data"
 #define API_KEY "YOUR_API_KEY"
 
 // Serial communication with Arduino
 #define RX_PIN D2
 #define TX_PIN D3
 SoftwareSerial arduinoSerial(RX_PIN, TX_PIN);
 
 // Buffer for incoming data
 String inputBuffer = "";
 bool dataReady = false;
 
 void setup() {
   // Initialize serial communication
   Serial.begin(9600);
   arduinoSerial.begin(9600);
   
   // Connect to WiFi
   WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
   
   Serial.print("Connecting to WiFi");
   while (WiFi.status() != WL_CONNECTED) {
     delay(500);
     Serial.print(".");
   }
   
   Serial.println();
   Serial.print("Connected to WiFi, IP address: ");
   Serial.println(WiFi.localIP());
 }
 
 void loop() {
   // Read data from Arduino
   while (arduinoSerial.available()) {
     char c = arduinoSerial.read();
     
     if (c == '\n') {
       dataReady = true;
     } else {
       inputBuffer += c;
     }
   }
   
   // Process and send data if ready
   if (dataReady) {
     Serial.println("Received data: " + inputBuffer);
     
     // Check if we have a valid JSON
     if (inputBuffer.startsWith("{") && inputBuffer.endsWith("}")) {
       sendDataToServer(inputBuffer);
     }
     
     // Reset buffer and flag
     inputBuffer = "";
     dataReady = false;
   }
   
   // Add delay to prevent overwhelming the server
   delay(100);
 }
 
 void sendDataToServer(String jsonData) {
   // Check WiFi connection
   if (WiFi.status() != WL_CONNECTED) {
     Serial.println("WiFi not connected");
     return;
   }
   
   // Add timestamp and device ID to the data
   DynamicJsonDocument doc(1024);
   deserializeJson(doc, jsonData);
   
   // Add timestamp (you could use NTP to get accurate time)
   doc["timestamp"] = millis();
   doc["device_id"] = ESP.getChipId();
   
   // Convert back to string
   String enhancedData;
   serializeJson(doc, enhancedData);
   
   // Create HTTP client
   HTTPClient http;
   WiFiClient client;
   
   // Begin HTTP request
   http.begin(client, SERVER_URL);
   
   // Add headers
   http.addHeader("Content-Type", "application/json");
   http.addHeader("X-API-Key", API_KEY);
   
   // Send POST request
   int httpResponseCode = http.POST(enhancedData);
   
   // Check response
   if (httpResponseCode > 0) {
     String response = http.getString();
     Serial.println("HTTP Response code: " + String(httpResponseCode));
     Serial.println("Response: " + response);
   } else {
     Serial.println("Error on sending POST: " + String(httpResponseCode));
   }
   
   // Close connection
   http.end();
 }