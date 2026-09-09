// Bhujaha (भुजः) — Shoulder Link (Link 1: Joint 2 to Joint 3)
include <parameters.scad>;

module shoulder_link() {
    difference() {
        union() {
            // Joint 2 bottom hub
            cylinder(r=20, h=16, center=true);

            // Arm truss beam (connecting Joint 2 to Joint 3)
            translate([L_SHOULDER_LINK / 2, 0, 0])
                cube([L_SHOULDER_LINK, 24, 16], center=true);

            // Joint 3 top hub
            translate([L_SHOULDER_LINK, 0, 0])
                cylinder(r=20, h=16, center=true);
        }

        // Weight reduction cutouts along the beam
        for (i = [1 : 3]) {
            translate([i * (L_SHOULDER_LINK / 4), 0, 0])
                rotate([0, 0, 45])
                    cube([18, 18, 20], center=true);
        }

        // Joint 2 servo horn mounting bore
        cylinder(r=SERVO_HORN_RADIUS + 1, h=18, center=true);
        for (a = [0 : 90 : 270]) {
            rotate([0, 0, a])
                translate([SERVO_HORN_RADIUS - 2, 0, 0])
                    cylinder(r=M3_HOLE / 2, h=20, center=true);
        }

        // Joint 3 servo mount cutout & bearing seat
        translate([L_SHOULDER_LINK, 0, 0]) {
            cylinder(r=BEARING_608_OD / 2, h=BEARING_608_WIDTH + 1, center=false);
            cylinder(r=BEARING_608_ID / 2, h=20, center=true);
        }

        // Cable routing slots
        translate([L_SHOULDER_LINK / 2, 0, 0])
            cube([L_SHOULDER_LINK - 40, 6, 20], center=true);
    }
}

shoulder_link();
