// Bhujaha (भुजः) — Forearm / Elbow Link (Link 2: Joint 3 to Wrist/Gripper)
include <parameters.scad>;

module elbow_link() {
    difference() {
        union() {
            // Joint 3 mounting hub
            cylinder(r=18, h=14, center=true);

            // Forearm beam
            translate([L_FOREARM_LINK / 2, 0, 0])
                cube([L_FOREARM_LINK, 20, 14], center=true);

            // End-effector mounting flange
            translate([L_FOREARM_LINK, 0, 0])
                cylinder(r=16, h=14, center=true);
        }

        // Weight reduction cutouts
        for (i = [1 : 3]) {
            translate([i * (L_FOREARM_LINK / 4), 0, 0])
                cylinder(r=7, h=16, center=true);
        }

        // Joint 3 servo horn interface
        cylinder(r=SERVO_HORN_RADIUS, h=16, center=true);
        for (a = [0 : 90 : 270]) {
            rotate([0, 0, a])
                translate([SERVO_HORN_RADIUS - 2, 0, 0])
                    cylinder(r=M3_HOLE / 2, h=16, center=true);
        }

        // Wrist / Gripper mounting holes (4x M3)
        translate([L_FOREARM_LINK, 0, 0]) {
            for (dx = [-8, 8]) {
                for (dy = [-8, 8]) {
                    translate([dx, dy, 0])
                        cylinder(r=M3_HOLE / 2, h=16, center=true);
                }
            }
        }
    }
}

elbow_link();
