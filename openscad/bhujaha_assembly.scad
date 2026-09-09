// Bhujaha (भुजः) — Complete 3-DOF Arm Assembly
// Interactive joint articulation variables (in degrees)
theta_1 = 30;   // Base yaw (-120 to +120)
theta_2 = 45;   // Shoulder pitch (-30 to +90)
theta_3 = -60;  // Elbow pitch (-120 to +60)
gripper_pos = 10; // Gripper open/close

include <parameters.scad>;
use <base.scad>;
use <shoulder_link.scad>;
use <elbow_link.scad>;
use <gripper.scad>;

// 1. Fixed Base Pedestal
color("DarkSlateGray") base_pedestal();

// 2. Joint 1: Turntable rotation (Yaw)
rotate([0, 0, theta_1]) {
    translate([0, 0, 25]) {
        color("SteelBlue") turntable_plate();

        // 3. Joint 2: Shoulder elevation (Pitch)
        translate([0, 0, L_BASE_HEIGHT - 25]) {
            rotate([0, theta_2, 0]) {
                color("RoyalBlue") shoulder_link();

                // 4. Joint 3: Elbow pitch
                translate([L_SHOULDER_LINK, 0, 0]) {
                    rotate([0, theta_3, 0]) {
                        color("DodgerBlue") elbow_link();

                        // 5. End Effector / Gripper
                        translate([L_FOREARM_LINK, 0, 0]) {
                            color("OrangeRed") gripper_base();
                            translate([12, 10, 8])
                                rotate([0, 0, -gripper_pos])
                                    color("Gold") gripper_finger();
                            translate([-12, 10, 8])
                                rotate([0, 0, gripper_pos])
                                    mirror([1, 0, 0])
                                        color("Gold") gripper_finger();
                        }
                    }
                }
            }
        }
    }
}
