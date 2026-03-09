import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from dataclasses import dataclass, field
from numpy import random as npr


GRAVITY = 9.81 # m/s^2
R = 8.314 # J/(mol*K)
BETA = 0.00367 # 1/K, volumetric expansion coefficient for air at room temperature
VIRTUAL_MASS = 0.5 # fraction of displaced air mass that contributes to inertia



@dataclass
class Bubble:
    spawn : np.array = field(default_factory=lambda: np.array([0.0, 0.0, 0.0])) # (x, y, time)
    base_temperature: float = 30.0
    idle_volume: float = 1.0
    
    def get_volume(self, ambient_temp):
        return self.idle_volume * (ambient_temp / self.base_temperature)
    
    def get_drag_acceleration(self, ambient_temp, w, radius):
        """
        Calculates deceleration due to air drag.
        F_drag = 0.5 * C_d * rho * Area * w^2
        a_drag = F_drag / m_effective
        """
        C_d = 0.47 # Drag coefficient for a sphere
        rho = 1.225 # kg/m^3 (approx air density)
        area = np.pi * radius**2
        m_eff = (rho * self.get_volume(ambient_temp)) * (1 + 0.5) # Mass + Virtual Mass
        
        drag_force = 0.5 * C_d * rho * area * (w**2)
        return drag_force / m_eff

    def get_lift_acceleration(self, ambient_temp):
        delta_temp = self.base_temperature - ambient_temp
        if delta_temp <= 0: return 0.0
        # Boussinesq + Virtual Mass
        return (GRAVITY * BETA * delta_temp) / (1 + VIRTUAL_MASS)

    def get_bubble_position_dynamic(self, time, a0, lambda_decay):
        """
        Stateless position calculation for a bubble with decaying buoyancy.

        a0: Initial buoyancy acceleration (m/s^2)
        lambda_decay: Rate of buoyancy loss (higher = faster cooling)
        """
        dt = time - self.spawn[2]  # Extract spawn time from the spawn array
        if dt <= 0:
            return 0.0

        # Position formula for exponentially decaying acceleration
        # z(t) = (a0 / lambda^2) * (lambda * t + exp(-lambda * t) - 1)
        # Refactored for numerical stability:
        z = (a0 / (lambda_decay**2)) * (lambda_decay * dt + np.exp(-lambda_decay * dt) - 1)

        return z
    def horizontal_translation(self, time, wind_vector):
        """
        Simple horizontal translation based on a constant wind vector.
        wind_vector: (vx, vy) in m/s
        """
        dt = time - self.spawn[2]
        if dt <= 0:
            return self.spawn[:2]  # No movement before spawn time
        return self.spawn[:2] + wind_vector * dt
    
    def get_position(self, time, a0, lambda_decay, wind_vector):
        z = self.get_bubble_position_dynamic(time, a0, lambda_decay)
        xy = self.horizontal_translation(time, wind_vector)
        return np.array([xy[0], xy[1], z])
    
    
@dataclass
class Spawner:
    
    spawn_rate: float
    temperature: float
    idle_volume: float
    spawn = np.array([0.0, 0.0, 0.0]) # (x, y, time) 
    bubbles: list = field(default_factory=list)

    
    def spawn_bubble(self, time):
        # Normal distribution around the spawn point
        pos = self.spawn[:2] + npr.normal(0, 1, size=2) * 10.0 # 10m stddev
        #clip to 500m radius
        if np.linalg.norm(pos - self.spawn[:2]) > 500.0:
            pos = self.spawn[:2] + (pos - self.spawn[:2]) / np.linalg.norm(pos - self.spawn[:2]) * 500.0
        
        bubble = Bubble(spawn=np.append(pos, time), base_temperature=self.temperature, idle_volume=self.idle_volume) 
        self.bubbles.append(bubble)
        return bubble
    
    
simulator = Spawner(spawn_rate=0.1, temperature=30.0, idle_volume=1.0)

for t in np.arange(0, 400, 1):
    number_to_spawn = np.random.poisson(simulator.spawn_rate)
    for _ in range(50):
        bubble = simulator.spawn_bubble(t)
        print(f"Spawned bubble at time {t:.1f} with base temp {bubble.base_temperature}°C and idle volume {bubble.idle_volume} m^3")
        
    
# at time t, get positions of all bubbles and plot them
time = 1000
wind_vector = np.array([1.0, 0.0]) # 1 m per second east, 0.5 m/s north
positions = np.array([bubble.get_position(time, a0=2.0, lambda_decay=0.05, wind_vector=wind_vector) for bubble in simulator.bubbles])

# plot slice at y=0
plt.figure(figsize=(10, 6))
plt.scatter(positions[:, 0], positions[:, 2], alpha=0.5)
plt.title(f"Bubble positions at time {time}s (y=0 slice)")
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.grid()
plt.show()


    

   
    
    
    
    


    
    