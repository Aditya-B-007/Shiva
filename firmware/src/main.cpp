#include <Arduino.h>
#include "servo_bus.h"
#include "trajectory_buffer.h"
#include "shiva_packet.h"

// Hardware Pin Definitions (ESP32)
#define PIN_SERVO_RX     16
#define PIN_SERVO_TX     17
#define PIN_GRIPPER_PWM  18
#define PIN_BTN_TEACH    25
#define PIN_BTN_PLAY     26
#define PIN_ESTOP        27
#define PIN_LED_STATUS   32
#define PIN_LED_RECORD   33

// Robot Operating States
enum RobotState {
    STATE_IDLE = 0,
    STATE_TEACHING,
    STATE_REPLAYING,
    STATE_ESTOP
};

volatile RobotState currentState = STATE_IDLE;
ServoBus servoBus(Serial2, PIN_SERVO_RX, PIN_SERVO_TX, 1000000);
TrajectoryBuffer trajectory;

unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_MS = 20; // 50 Hz control loop

void IRAM_ATTR handleEmergencyStop() {
    currentState = STATE_ESTOP;
}

void setup() {
    // Host USB / UART interface to Shiva runtime
    Serial.begin(115200);

    // Servo Bus
    servoBus.begin();

    // Pin Configurations
    pinMode(PIN_BTN_TEACH, INPUT_PULLUP);
    pinMode(PIN_BTN_PLAY, INPUT_PULLUP);
    pinMode(PIN_ESTOP, INPUT_PULLUP);
    pinMode(PIN_LED_STATUS, OUTPUT);
    pinMode(PIN_LED_RECORD, OUTPUT);

    attachInterrupt(digitalPinToInterrupt(PIN_ESTOP), handleEmergencyStop, FALLING);

    // Initial State
    servoBus.setAllTorque(true);
    digitalWrite(PIN_LED_STATUS, HIGH);
    digitalWrite(PIN_LED_RECORD, LOW);
}

void loop() {
    unsigned long now = millis();

    // 1. Emergency Stop Check
    if (currentState == STATE_ESTOP || digitalRead(PIN_ESTOP) == LOW) {
        currentState = STATE_ESTOP;
        servoBus.setAllTorque(false);
        digitalWrite(PIN_LED_STATUS, LOW);
        digitalWrite(PIN_LED_RECORD, HIGH);
        return; // Halt all operations
    }

    // 2. Physical Button Handling
    if (digitalRead(PIN_BTN_TEACH) == LOW) {
        delay(50); // Simple debounce
        if (digitalRead(PIN_BTN_TEACH) == LOW) {
            if (currentState != STATE_TEACHING) {
                // Enter compliant teaching mode
                currentState = STATE_TEACHING;
                trajectory.clear();
                servoBus.setAllTorque(false); // Unpower motors so user can move arm
                digitalWrite(PIN_LED_RECORD, HIGH);
            } else {
                // Stop teaching, lock arm
                currentState = STATE_IDLE;
                servoBus.setAllTorque(true);
                digitalWrite(PIN_LED_RECORD, LOW);
            }
            while (digitalRead(PIN_BTN_TEACH) == LOW); // Wait for release
        }
    }

    if (digitalRead(PIN_BTN_PLAY) == LOW) {
        delay(50);
        if (digitalRead(PIN_BTN_PLAY) == LOW) {
            if (currentState != STATE_REPLAYING && trajectory.count() > 0) {
                currentState = STATE_REPLAYING;
                trajectory.resetPlayback();
                servoBus.setAllTorque(true); // Energize motors to hold/move
            } else {
                currentState = STATE_IDLE;
            }
            while (digitalRead(PIN_BTN_PLAY) == LOW);
        }
    }

    // 3. Periodic Execution (50 Hz Control Loop)
    if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        lastSampleTime = now;

        float currentTheta[3] = {0.0f, 0.0f, 0.0f};
        float currentAmps[3] = {0.0f, 0.0f, 0.0f};

        // Query positions and currents from all 3 joint magnetic encoders
        for (uint8_t id = 1; id <= 3; id++) {
            servoBus.readPosition(id, currentTheta[id - 1]);
            servoBus.readCurrent(id, currentAmps[id - 1]);
        }

        // State Machine Execution
        if (currentState == STATE_TEACHING) {
            // Record joint angles into memory buffer
            trajectory.addPoint(currentTheta[0], currentTheta[1], currentTheta[2], 0.0f);
        } else if (currentState == STATE_REPLAYING) {
            // Replay recorded points
            TrajectoryPoint pt;
            if (trajectory.getNextPoint(pt)) {
                for (uint8_t id = 1; id <= 3; id++) {
                    servoBus.writePosition(id, pt.theta[id - 1], SAMPLE_INTERVAL_MS, 0);
                }
            }
        }

        // 4. Stream Telemetry to Shiva Runtime via USB Serial
        TelemetryPacket pkt;
        pkt.header = SHIVA_MAGIC_HEADER;
        pkt.timestamp_ms = now;
        pkt.theta[0] = currentTheta[0];
        pkt.theta[1] = currentTheta[1];
        pkt.theta[2] = currentTheta[2];
        pkt.current[0] = currentAmps[0];
        pkt.current[1] = currentAmps[1];
        pkt.current[2] = currentAmps[2];
        pkt.gripper_pos = 0.0f;
        pkt.system_state = (uint8_t)currentState;
        pkt.checksum = 0; // Checksum computation
        Serial.write((const uint8_t*)&pkt, sizeof(pkt));
    }

    // 5. Ingest Commands from Host Shiva Runtime (if available)
    if (Serial.available() >= (int)sizeof(CommandPacket)) {
        CommandPacket cmd;
        Serial.readBytes((char*)&cmd, sizeof(cmd));
        if (cmd.header == SHIVA_MAGIC_HEADER && currentState != STATE_TEACHING && currentState != STATE_ESTOP) {
            // Execute setpoints commanded by Shiva's CPO safety consensus pipeline
            for (uint8_t id = 1; id <= 3; id++) {
                servoBus.writePosition(id, cmd.target_theta[id - 1], cmd.execution_time, 0);
            }
        }
    }
}
