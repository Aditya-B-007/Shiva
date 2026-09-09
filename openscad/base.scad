// Bhujaha (भुजः) — Base & Turntable (Joint 1 / Yaw)
include <parameters.scad>;

module base_pedestal() {
    difference() {
        // Main base body
        cylinder(r=60, h=25, center=false);

        // Center cutout for wiring and bearing seat
        translate([0, 0, 10])
            cylinder(r=BEARING_608_OD / 2, h=16, center=false);

        translate([0, 0, -1])
            cylinder(r=BEARING_608_ID / 2 + 1, h=12, center=false);

        // Table mounting bolt holes (4x M4 at 90 deg)
        for (a = [0, 90, 180, 270]) {
            rotate([0, 0, a])
                translate([45, 0, -1])
                    cylinder(r=M4_HOLE / 2, h=30, center=false);
        }

        // Cable pass-through channel
        translate([0, 0, 5])
            cube([70, 12, 10], center=true);
    }
}

module turntable_plate() {
    difference() {
        union() {
            // Rotating plate
            cylinder(r=55, h=8, center=false);

            // Bearing hub
            translate([0, 0, -BEARING_608_WIDTH])
                cylinder(r=BEARING_608_ID / 2 - 0.1, h=BEARING_608_WIDTH, center=false);
        }

        // Center wire bore
        translate([0, 0, -10])
            cylinder(r=5, h=30, center=false);

        // Servo bracket mounting holes for Joint 1
        for (dx = [-SERVO_MOUNT_SPACING_X/2, SERVO_MOUNT_SPACING_X/2]) {
            for (dy = [-SERVO_MOUNT_SPACING_Y/2, SERVO_MOUNT_SPACING_Y/2]) {
                translate([dx, dy, -1])
                    cylinder(r=M3_HOLE / 2, h=15, center=false);
            }
        }
    }
}

// Standalone render
base_pedestal();
translate([0, 0, 35]) turntable_plate();
