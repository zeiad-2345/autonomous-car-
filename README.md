# Autonomous Car Project — BFMC

This project is an autonomous mini-car developed for the Bosch Future Mobility Challenge (BFMC).  
The car is designed to drive inside a smart-city environment while following lanes, detecting road signs, detecting pedestrians, and responding to traffic lights.

## Current Features

- Lane keeping and lane following
- Pedestrian detection
- Traffic sign detection
- Traffic light detection
- Real-time camera processing
- Raspberry Pi and Arduino integration
- Motor speed control and steering control
- Autonomous driving inside a smart-city track

## Demo Videos

### Lane Keeping Demo

[Watch Lane Keeping Demo](PUT_VIDEO_LINK_HERE)

### Detection Demo

[Watch Sign / Pedestrian / Traffic Light Detection Demo](PUT_VIDEO_LINK_HERE)

## Project Overview

The system is divided into multiple layers:

### High-Level Processing

The high-level system handles computer vision and decision making.  
It processes camera frames and detects important objects in the environment, such as:

- pedestrians
- traffic signs
- traffic lights
- lane markings

### Low-Level Control

The low-level system controls the physical movement of the car using Arduino.  
It receives commands for:

- steering angle
- motor direction
- motor speed
- stopping and braking

### Hardware Used

- Raspberry Pi
- Arduino
- Camera module / USB camera
- DC motor
- Servo motor
- Motor driver
- 3D printed car parts
- Custom wiring and power system

## Technologies Used

- Python
- OpenCV
- YOLO / object detection model
- Arduino C++
- Serial communication
- Raspberry Pi
- Git / GitHub

## My Contribution

I worked on the autonomous driving system, including:

- lane keeping logic
- object detection integration
- communication between the Raspberry Pi / laptop and Arduino
- motor and steering control
- testing and debugging the car behavior on track

## Project Status

The car is currently able to keep lanes and detect pedestrians, signs, and traffic lights.  
Further improvements are being made to increase stability, reduce steering oscillations, and improve real-time performance.

## Future Improvements

- Improve lane keeping stability
- Reduce steering oscillations
- Improve detection accuracy
- Optimize FPS and latency
- Add better decision-making for intersections
- Improve traffic light handling
- Improve recovery when the car loses the lane

