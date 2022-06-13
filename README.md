# Total Energy Shaping with Neural Interconnection and Damping Assignment - Passivity Based Control (IDA-PBC)
This repository contains the implementation of the methodology described in:

Santiago Sanchez-Escalonilla et al., [Total Energy Shaping with Neural Interconnection and Damping Assignment - Passivity Based Control](https://arxiv.org/abs/2112.12999), Accepted in 4th Annual Learning for Dynamics and Control Conference, July 2022

## What is energy shaping?
Energy shaping allows to exploit the real dynamics of complex systems.
![Energy shaping example](./figures/energy_poster2.svg)

The energy function of the simple pendulum, $H(x)$ (**left**),shows multiple equilibria. IDA-PBC's main objective is to shape the closed loop energy, $H_d(x)$ (**right**), such that the closed loop dynamics are stable around a desired point $x^\star$.

Neural IDA-PBC enables the use of this popular nonlinear control technique to cases were finding an analytical solution that satisfies both the matching equations and structural constraints would be troublesome.

## How to use this repository
This repository contains two working examples of mechanical systems:
1. [Simple pendulum](https://github.com/ssantus/L4DC-Neural-IDAPBC/blob/main/simple_pendulum/simple_pendulum_main.py)
2. [Double pendulum](https://github.com/ssantus/L4DC-Neural-IDAPBC/blob/main/double_pendulum/double_pendulum_main.py)

To test the methodology you can directly navigate to the main file of either one of the examples ([1](https://github.com/ssantus/L4DC-Neural-IDAPBC/blob/main/simple_pendulum/simple_pendulum_main.py), [2](https://github.com/ssantus/L4DC-Neural-IDAPBC/blob/main/double_pendulum/double_pendulum_main.py)). This main file can be used for either training or simulating the time response.

## Requirements
This repository runs on Python 3.9, tensorflow 2.6, numpy 1.19.5 and matplotlib 3.4.3.
