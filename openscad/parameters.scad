// Bhujaha (भुजः) — Global CAD Parameters
// All units in millimeters (mm)

$fn = 60; // Curve resolution

// Joint Link Lengths
L_BASE_HEIGHT     = 65.0;   // Ground to Shoulder joint axis
L_SHOULDER_LINK   = 180.0;  // Shoulder to Elbow axis
L_FOREARM_LINK    = 180.0;  // Elbow to Wrist/Gripper mount
L_GRIPPER_REACH   = 60.0;   // Gripper reach

// Hardware Dimensions (Standard Off-The-Shelf)
BEARING_608_OD    = 22.2;   // 608RS outer diameter (with tolerance)
BEARING_608_ID    = 8.0;    // 608RS inner bore
BEARING_608_WIDTH = 7.0;    // 608RS thickness

// Fasteners
M3_HOLE           = 3.4;    // Clearance for M3 screw
M3_NUT_HEX        = 6.2;    // M3 hex nut slot width
M4_HOLE           = 4.4;    // Clearance for M4 screw
M4_NUT_HEX        = 7.8;    // M4 hex nut slot width

// Serial Bus Servo Specs (Feetech STS3215 / SCS15 standard envelope)
SERVO_LENGTH      = 45.2;
SERVO_WIDTH       = 24.7;
SERVO_HEIGHT      = 35.0;
SERVO_HORN_RADIUS = 10.0;
SERVO_MOUNT_SPACING_X = 48.5;
SERVO_MOUNT_SPACING_Y = 10.0;

// Wall Thickness & Tolerances
WALL_THICKNESS    = 4.0;
TOLERANCE         = 0.3;    // 3D printer clearance
