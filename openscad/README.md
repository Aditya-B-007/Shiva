# Bhujaha (भुजः) — 3D Printing & Mechanical Specifications

This directory contains parametric [OpenSCAD](https://openscad.org/) CAD source files for manufacturing the Bhujaha 3-DOF robotic manipulator.

## File Map
* `parameters.scad`: Global arm dimensions, link lengths, tolerances, and hardware specifications.
* `base.scad`: Base pedestal and Joint 1 (Yaw) rotating turntable with 608RS bearing seat.
* `shoulder_link.scad`: Heavy-duty upper arm link (Joint 2 to Joint 3, 180 mm axis-to-axis).
* `elbow_link.scad`: Forearm link (Joint 3 to Wrist/Gripper mount, 180 mm axis-to-axis).
* `gripper.scad`: Parallel-jaw gripper driven by an MG90S micro servo.
* `bhujaha_assembly.scad`: Full 3-DOF kinematic assembly with adjustable joint angles (`theta_1`, `theta_2`, `theta_3`).

## Recommended 3D Print Settings
* **Material**: PETG (Recommended for rigidity and layer adhesion) or tough PLA+
* **Infill**: 40%–50% Gyroid or Honeycomb (critical for structural links to avoid flex under payload)
* **Perimeters / Walls**: 4 to 5 walls
* **Top/Bottom Layers**: 5 layers
* **Nozzle**: 0.4 mm or 0.6 mm
* **Layer Height**: 0.2 mm

## Required Mechanical Hardware
* 3× 608RS Deep Groove Ball Bearings (8x22x7 mm)
* 10× M3 × 12 mm socket head screws & M3 lock nuts
* 6× M3 × 16 mm socket head screws
* 4× M4 × 20 mm bolts (for table clamp / base mounting)
* 3× Feetech STS3215 / SCS15 standard aluminum servo horns
