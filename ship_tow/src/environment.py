import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MultiAgentShipTowEnv(gym.Env):
    def __init__(self,
                grid_size=400,
                dock_position=(200, 350),
                dock_dim=(100, 50),
                target_position=(250, 400),
                frame_update=0.01,
                ship_dim=(60, 8),
                ship_mass=6.0,
                ship_inertia=0.1,
                ship_velocity=0.025,
                ship_angular_velocity=0.001,
                tugboat_dim=(5, 2),
                max_rope_length=10.0,
                linear_drag_coeff=1.0,
                angular_drag_coeff=3.0
                ):
        super().__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.dock_position = dock_position
        self.dock_dim = dock_dim
        self.target_position = target_position
        self.dt = frame_update
        
        # Ship parameters
        self.ship_dim = ship_dim
        self.ship_mass = ship_mass
        self.ship_inertia = ship_inertia
        self.ship_velocity = np.array([ship_velocity, ship_velocity])
        self.angular_velocity = ship_angular_velocity
        
        # Tugboat parameters
        self.tugboat_dim = tugboat_dim
        self.max_rope_length = max_rope_length
        self.front_offset = ship_dim[0]
        self.z = np.sqrt((self.ship_dim[0]**2) + (self.ship_dim[1]**2))
        
        # Water drag
        self.linear_drag_coeff = linear_drag_coeff
        self.angular_drag_coeff = angular_drag_coeff

        # for each agent (tugboat)
        self.action_space = {
            'tugboat_1': spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([15.0, 15.0]),
                dtype=np.float32
            ),
            'tugboat_2': spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([15.0, 15.0]),
                dtype=np.float32
            )
        }


        self.observation_space = {
            'tugboat_1': spaces.Box(
                low=np.array([
                        0.0,                    # min x position (ship)
                        0.0,                    # min y position (ship)
                        -np.pi,                 # min rotation (ship)
                        0.0,                    # min x position (own tugboat)
                        0.0,                    # min y position (own tugboat)
                        -np.pi,                 # min rotation (own tugboat)
                        0.0,                    # min x position (other tugboat)
                        0.0,                    # min y position (other tugboat)
                        -np.pi,                  # min rotation (other tugboat)
                        0.5,                    # min distance to target
                        0.0                     # min rope length
                            ], dtype=np.float32),
                high=np.array([
                        self.grid_size,         # max x position (ship)
                        self.grid_size,         # max y position (ship)
                        np.pi,                  # max rotation (ship)
                        self.grid_size,         # max x position (own tugboat)
                        self.grid_size,         # max y position (own tugboat)
                        np.pi,                  # max rotation (own tugboat)
                        self.grid_size,         # max x position (other tugboat)
                        self.grid_size,         # max y position (other tugboat)
                        np.pi,                  # max rotation (other tugboat)
                        np.sqrt(2)*self.grid_size, # max possible distance (diagonal)
                        self.max_rope_length    # max rope length
                            ], dtype=np.float32)),
            'tugboat_2': spaces.Box(
                low=np.array([
                        0.0,                    # min x position (ship)
                        0.0,                    # min y position (ship)
                        -np.pi,                 # min rotation (ship)
                        0.0,                    # min x position (own tugboat)
                        0.0,                    # min y position (own tugboat)
                        -np.pi,                 # min rotation (own tugboat)
                        0.0,                    # min x position (other tugboat)
                        0.0,                    # min y position (other tugboat)
                        -np.pi,                  # min rotation (other tugboat)
                        0.5,                    # min distance to target
                        0.0                     # min rope length
                            ], dtype=np.float32),
                high=np.array([
                        self.grid_size,         # max x position (ship)
                        self.grid_size,         # max y position (ship)
                        np.pi,                  # max rotation (ship)
                        self.grid_size,         # max x position (own tugboat)
                        self.grid_size,         # max y position (own tugboat)
                        np.pi,                  # max rotation (own tugboat)
                        self.grid_size,         # max x position (other tugboat)
                        self.grid_size,         # max y position (other tugboat)
                        np.pi,                  # max rotation (other tugboat)
                        np.sqrt(2)*self.grid_size, # max possible distance (diagonal)
                        self.max_rope_length    # max rope length
                            ], dtype=np.float32)
            )
        }

        self.obstacles = [(0, 100, 200, 50)]  # (x, y, width, height)
        


    def reset(self, seed=None):
        super().reset(seed=seed)
        
        ship_state = np.array([
            50.0, 50.0, 0.0,  # Ship position and orientation
        ])
        
        tugboat1_state = np.array([
            55.0 + self.front_offset, 50.0, 0.0,  # pos and orientation
        ])
        
        tugboat2_state = np.array([
            55.0 + self.front_offset, 50.0 + self.ship_dim[1], 0.0,  # pos and orientation
        ])
        
        self.state = np.concatenate([
            ship_state,
            tugboat1_state,
            tugboat2_state,
            [np.sqrt((ship_state[0] - self.target_position[0])**2 + (ship_state[1] - self.target_position[1]))],  # Distance to target
            [self.max_rope_length]
        ])
        
        return self._get_observations()
    
    

    def _get_observations(self):
        """Returns observations for each agent."""
        xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l = self.state
        
        obs_tugboat1 = np.array([
            xs, ys, thetas,          # Ship state
            xt1, yt1, thetat1,       # Own state
            xt2, yt2, thetat2,         # Other tugboat state
            ds,                   # Distance to target
            np.linalg.norm(np.array([xt1, yt1]) - np.array([xs, ys]))  # rope length
        ])
        
        obs_tugboat2 = np.array([
            xs, ys, thetas,          # Ship state
            xt2, yt2, thetat2,       # Own state
            xt1, yt1, thetat1,         # Other tugboat state
            ds,                   # Distance to target
            np.linalg.norm(np.array([xt2, yt2]) - np.array([xs, ys]))  # Own reward
        ])
        
        return {
            'tugboat_1': obs_tugboat1,
            'tugboat_2': obs_tugboat2
        }
    

    
    def check_collision(self, x, y, length, breadth, object_type='ship'):
        for (ox, oy, owidth, oheight) in self.obstacles:
            if (x < ox + owidth and x + length > ox and
                y < oy + oheight and y + breadth > oy):
                return True
        
        
        if object_type != 'ship':
            ship_x, ship_y = self.state[0], self.state[1]
            if (x < ship_x + self.ship_dim[0]/2 and x + length > ship_x - self.ship_dim[0]/2 and
                y < ship_y + self.ship_dim[1]/2 and y + breadth > ship_y - self.ship_dim[1]/2):
                return True
        
        return False



    # def _get_reward(self, agent_id):
    #     """Calculate reward for specific agent."""
    #     xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l = self.state
        
    #     # Base reward based on progress toward target
    #     reward = -ds / self.grid_size
        
    #     # Penalty for stretching rope too much
    #     if agent_id == 'tugboat_1':
    #         rope_length = np.linalg.norm(np.array([xt1, yt1]) - np.array([xs, ys]))
    #     else:
    #         rope_length = np.linalg.norm(np.array([xt2, yt2]) - np.array([xs, ys]))
            
    #     if rope_length > self.max_rope_length:
    #         reward -= (rope_length - self.max_rope_length)
        
    #     # Success reward
    #     if ds < 5.0 and abs(thetas) < 0.1:
    #         reward += 100.0
            
    #     # Collision penalty
    #     if self.check_collision(xs, ys, self.ship_dim[0], self.ship_dim[1], 'ship'):
    #         reward -= 50.0
            
    #     return reward

    def calculate_distance_to_obstacle(self, x, y, obj_width, obj_height):
        """Calculate minimum distance from an object to any obstacle."""
        min_distance = float('inf')
        
        # object corners
        corners = [
            (x, y),  # Top-left
            (x + obj_width, y),  # Top-right
            (x, y + obj_height),  # Bottom-left
            (x + obj_width, y + obj_height)  # Bottom-right
        ]
        
        for obstacle in self.obstacles:
            ox, oy, ow, oh = obstacle
            
            # obstacle corners
            obs_corners = [
                (ox, oy),  # Top-left
                (ox + ow, oy),  # Top-right
                (ox, oy + oh),  # Bottom-left
                (ox + ow, oy + oh)  # Bottom-right
            ]
            
            # Check if object is inside obstacle
            if (x < ox + ow and x + obj_width > ox and
                y < oy + oh and y + obj_height > oy):
                return 0.0
            
            # Calculate minimum distance from any corner to obstacle edges
            for cx, cy in corners:
                # Calculate distances to obstacle edges
                dx = max(ox - cx, 0, cx - (ox + ow))
                dy = max(oy - cy, 0, cy - (oy + oh))
                dist = np.sqrt(dx * dx + dy * dy)
                min_distance = min(min_distance, dist)
                
            # Calculate minimum distance from obstacle corners to object edges
            for cx, cy in obs_corners:
                dx = max(x - cx, 0, cx - (x + obj_width))
                dy = max(y - cy, 0, cy - (y + obj_height))
                dist = np.sqrt(dx * dx + dy * dy)
                min_distance = min(min_distance, dist)
        
        return min_distance
    
    # Proximity from obstacles
    def calculate_proximity_penalty(self, distance, danger_zone=40.0, max_penalty=50.0):
        if distance >= danger_zone:
            return 0.0
        
        # Exponentially increasing penalty as distance decreases
        penalty_factor = ((danger_zone - distance) / danger_zone) ** 2
        return max_penalty * penalty_factor


    def _get_reward(self, agent_id):
        xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l = self.state
        
        # Base reward based on progress toward target
        reward = -ds / self.grid_size
        
        # Calculate proximity penalties
        ship_distance = self.calculate_distance_to_obstacle(
            xs, ys, self.ship_dim[0], self.ship_dim[1]
        )
        ship_penalty = self.calculate_proximity_penalty(ship_distance)
        
        # Calculate tugboat penalties
        if agent_id == 'tugboat_1':
            tug_pos = (xt1, yt1)
        else:
            tug_pos = (xt2, yt2)
            
        tug_distance = self.calculate_distance_to_obstacle(
            tug_pos[0], tug_pos[1], self.tugboat_dim[0], self.tugboat_dim[1]
        )
        tug_penalty = self.calculate_proximity_penalty(tug_distance)
        
        # Apply penalties
        reward -= (ship_penalty + tug_penalty)
        
        # Penalty for stretching rope too much
        if agent_id == 'tugboat_1':
            rope_length = np.linalg.norm(np.array([xt1, yt1]) - np.array([xs, ys]))
        else:
            rope_length = np.linalg.norm(np.array([xt2, yt2]) - np.array([xs, ys]))
            
        if rope_length > self.max_rope_length:
            reward -= (rope_length - self.max_rope_length)
        
        # Success reward
        if ds < 6.0 and abs(thetas) < 0.1:
            reward += 1000.0
            
        # Immediate collision penalty
        # if self.check_collision(xs, ys, self.ship_dim[0], self.ship_dim[1], 'ship'): # replace agent_id with ship
        #     reward -= 100.0
        if tug_distance < 1:
            reward -= 100
            
        return reward
    

    def step(self, actions):
        
        vx1, vy1 = actions['tugboat_1']
        vx2, vy2 = actions['tugboat_2']
        
       
        xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l = self.state

        # Attachment points
        P_fr = np.array([xs + self.front_offset * np.cos(thetas),
                        ys + self.front_offset * np.sin(thetas)])
        P_fl = np.array([xs + self.z*np.cos(thetas+np.arctan(self.ship_dim[1]/self.ship_dim[0])),
                        ys + self.z*np.sin(thetas+np.arctan(self.ship_dim[1]/self.ship_dim[0]))])

        xt1 += vx1 * self.dt
        yt1 += vy1 * self.dt
        xt2 += vx2 * self.dt
        yt2 += vy2 * self.dt

        rope1_vector = np.array([xt1, yt1]) - P_fr
        rope2_vector = np.array([xt2, yt2]) - P_fl

        rope1_length = np.linalg.norm(rope1_vector)
        rope2_length = np.linalg.norm(rope2_vector)

        if rope1_length > self.max_rope_length:
            xt1, yt1 = P_fr + (rope1_vector / rope1_length) * self.max_rope_length

        if rope2_length > self.max_rope_length:
            xt2, yt2 = P_fl + (rope2_vector / rope2_length) * self.max_rope_length

        force_front_1 = (rope1_vector / self.max_rope_length) * 2.0
        force_front_2 = (rope2_vector / self.max_rope_length) * 2.0

        net_force = force_front_1 + force_front_2
        linear_drag_force = -self.linear_drag_coeff * self.ship_velocity
        net_force += linear_drag_force

        acceleration = net_force / self.ship_mass
        self.ship_velocity += acceleration * self.dt
        xs += self.ship_velocity[0] * self.dt
        ys += self.ship_velocity[1] * self.dt

        torque = np.cross(P_fr - np.array([xs, ys]), force_front_1 + force_front_2)
        angular_drag_torque = -self.angular_drag_coeff * self.angular_velocity
        torque += angular_drag_torque

        angular_acceleration = torque / self.ship_inertia
        self.angular_velocity += angular_acceleration * self.dt
        thetas += self.angular_velocity * self.dt

        # Update distance to target
        new_ds = np.linalg.norm(np.array([xs+self.front_offset, ys+self.front_offset]) - np.array(self.target_position))

        # Update state
        self.state = np.array([xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, new_ds, l])

        # Get observations and rewards
        observations = self._get_observations()
        rewards = {
            'tugboat_1': self._get_reward('tugboat_1'),
            'tugboat_2': self._get_reward('tugboat_2')
        }

        done = False
        
        # Check termination
        done_tugboat_1 = self.check_collision(xt1, yt1, self.tugboat_dim[0], self.tugboat_dim[1], 'tugboat')
        done_tugboat_2 = self.check_collision(xt2, yt2, self.tugboat_dim[0], self.tugboat_dim[1], 'tugboat')
        done_ship = new_ds < 5.0 or self.check_collision(xs, ys, self.ship_dim[0], self.ship_dim[1], 'ship')
        if done_tugboat_1 or done_tugboat_2 or done_ship:
            done = True
        dones = {
            'tugboat_1': done,
            'tugboat_2': done,
            '__all__': done
        }

        return observations, rewards, dones, {}


    def render(self):
        """Render the environment."""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()
        else:
            self.ax.clear()

        self.ax.set_facecolor('lightblue')
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_xticks(np.arange(0, self.grid_size+1, 50))  # 10 unit spacing for X-axis
        self.ax.set_yticks(np.arange(0, self.grid_size+1, 50))  # 10 unit spacing for Y-axis
        self.ax.grid(which='both', linestyle='-', linewidth=0.5, color='gray')

        # Draw dock
        dock_patch = patches.Rectangle(
            xy=self.dock_position,
            width=self.dock_dim[0],
            height=self.dock_dim[1],
            edgecolor='brown',
            facecolor='sienna'
        )
        self.ax.add_patch(dock_patch)

        # Unpack state
        xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l = self.state

        # Draw ship
        ship_patch = patches.Rectangle(
            # (xs - self.ship_dim[0], ys - self.ship_dim[1]),
            (xs, ys),
            self.ship_dim[0],
            self.ship_dim[1],
            angle=np.degrees(thetas),
            edgecolor='grey',
            facecolor='grey'
        )
        self.ax.add_patch(ship_patch)

        # Draw tugboats
        tugboat1_patch = patches.Rectangle(
            (xt1, yt1),
            self.tugboat_dim[0],
            self.tugboat_dim[1],
            angle=np.degrees(thetas),
            edgecolor='green',
            facecolor='green'
        )
        tugboat2_patch = patches.Rectangle(
            (xt2 , yt2),
            self.tugboat_dim[0],
            self.tugboat_dim[1],
            angle=np.degrees(thetas),
            edgecolor='orange',
            facecolor='yellow'
        )
        self.ax.add_patch(tugboat1_patch)
        self.ax.add_patch(tugboat2_patch)

        # Draw ropes
        P_fr = np.array([xs + self.front_offset * np.cos(thetas),
                        ys + self.front_offset * np.sin(thetas)])
        P_fl = np.array([xs + self.z*np.cos(thetas+np.arctan(self.ship_dim[1]/self.ship_dim[0])),
                        ys + self.z*np.sin(thetas+np.arctan(self.ship_dim[1]/self.ship_dim[0]))])

        self.ax.plot([P_fr[0], xt1], [P_fr[1], yt1], 'k-', lw=1)
        self.ax.plot([P_fl[0], xt2], [P_fl[1], yt2], 'k-', lw=1)

        # Draw obstacles
        for (ox, oy, owidth, oheight) in self.obstacles:
            obs_patch = patches.Rectangle(
                (ox, oy),
                width=owidth,
                height=oheight,
                edgecolor='red',
                facecolor='lightcoral'
            )
            self.ax.add_patch(obs_patch)

        plt.pause(0.001)
        plt.draw()

# if __name__ == "__main__":
#     # Create environment
#     env = MultiAgentShipTowEnv()
    
#     observations = env.reset()
#     done = False

#     # Get target position from environment
#     target_position = env.target_position

#     while not done:
#         actions = {
#             'tugboat_1': env.action_space['tugboat_1'].sample(),
#             'tugboat_2': env.action_space['tugboat_2'].sample()
#         }

#         observations, rewards, dones, _ = env.step(actions)
#         print(f"Rewards: {rewards}")
#         ship_x = observations['tugboat_1'][0]  

#         if ship_x > target_position[0] or dones['__all__']:
#             done = True
#         env.render()

#     env.close()
#     plt.close('all')