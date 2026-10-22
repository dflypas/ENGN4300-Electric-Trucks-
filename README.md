# DairyDrive Electric – ENGN4300

## Project Overview

DairyDrive Electric is an ANU ENGN4300 Capstone Design Project investigating the electrification of a small-scale milk delivery fleet.

The project developed a mixed-integer linear programming (MILP) optimisation model to determine the most cost-effective combination of electric and diesel trucks, batteries, charging infrastructure, renewable energy generation and energy storage over a ten-year analysis period.

The model determines the optimal, feasible implementation, considering capital investment, operational and maintenance costs, replacement costs and residual asset values, while meeting all fleet delivery and energy requirements.

## Repository Structure

| Folder | Description |
|---|---|
| **Optimisation Model** | Main optimisation model, supporting modules, input files. |
| **Diesel Price Prediction Model** | Code and resources used to predict diesel fuel prices. |
| **Spot Price Prediction Model** | Code and resources used to predict electricity spot prices. |
| **Version Control and Other Documents** | Previous model versions and old development files. |

## Optimisation Model

The optimisation considers the following system components:

- Diesel Delivery trucks
- Electric retrofit trucks
- Swappable batteries and charging stations
- Stationary battery energy storage systems (BESS)
- On-site solar photovoltaic (PV) generation
- Grid electricity import and export
- Delivery route scheduling and energy requirements

The model is developed in Python using the Gurobi optimisation solver.

## Project Team

Developed as part of the Australian National University ENGN4300 Capstone Design Project, 2026.

**Team members:** Daniel Parton, Jahin Jamil, Zarif Karim, Joel Saunders, Declan Pascoe and Molly Robertson.

## Repository Purpose

This repository contains the project's optimisation code and necessary supporting prediction models. 
