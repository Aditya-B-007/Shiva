// Bhujaha (भुजः) — End-Effector / Micro Parallel Jaw Gripper
include <parameters.scad>;

// Micro servo (MG90S / SG90) dimensions
SERVO_MICRO_L = 22.8;
SERVO_MICRO_W = 12.2;
SERVO_MICRO_H = 28.5;

module gripper_base() {
    difference() {
        // Main gripper bracket body
        cube([34, 30, 24], center=true);

        // Micro servo pocket
        translate([0, 0, 2])
            cube([SERVO_MICRO_L + 0.5, SERVO_MICRO_W + 0.5, SERVO_MICRO_H], center=true);

        // Arm attachment bolt holes
        for (dx = [-8, 8]) {
            for (dy = [-8, 8]) {
                translate([dx, dy, -10])
                    cylinder(r=M3_HOLE / 2, h=10, center=true);
            }
        }

        // Jaw pivot axle holes
        translate([12, 10, 0]) cylinder(r=M3_HOLE / 2, h=30, center=true);
        translate([-12, 10, 0]) cylinder(r=M3_HOLE / 2, h=30, center=true);
    }
}

module gripper_finger() {
    difference() {
        union() {
            // Pivot hub
            cylinder(r=6, h=6, center=true);

            // Finger tip with compliant grip surface
            translate([0, 25, 0])
                cube([8, 50, 6], center=true);

            // Grip pad
            translate([3, 45, 0])
                cube([4, 12, 6], center=true);
        }

        // Pivot bolt hole
        cylinder(r=M3_HOLE / 2, h=10, center=true);
    }
}

// Standalone render
gripper_base();
translate([12, 10, 8]) gripper_finger();
translate([-12, 10, 8]) mirror([1, 0, 0]) gripper_finger();
