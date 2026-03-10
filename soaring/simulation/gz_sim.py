import probability_map
import numpy as np
import probability_map
from probability_map import plot_sources
from thermal_field import ThermalField
from thermal_model import ThermalState


rng = np.random.default_rng(42)
map_key = 'p_spawn'

mp = probability_map.load_tif('../data/probability_maps/val_de_ruz_may_avg.tif')

#rate = 0.04*min*km^2

km2 = mp[map_key].shape[0] * mp[map_key].shape[1] * (mp['resolution'] ** 2) / 1e6

prior = mp[map_key]*km2*0.04/60
print(f"Map area: {km2:.2f} km^2")
print(f"Prior spawn rate: {prior.sum()}")

time = 0.0
spawn_time = 0.0
# ask gazebo for timing;

tfield = ThermalField(mp, z_i=1200.0, spawn_rate=1e-4, rng=rng)

wind = np.array([2.0, 0.5])

while 1:
    gazebo_time = 0.0 # get gazebo time
    tfield.update(gazebo_time,wind)
                
            
            
    
        
    
    

