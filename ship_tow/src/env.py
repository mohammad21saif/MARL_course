import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ShipTowEnv(gym.Env):
    def __init__(self,
                grid_size,
                dock_position, 
                target_position,
                frame_update,
                ship_dim, 
                ship_mass, 
                ship_inertia,
                ship_velocity,
                ship_angular_velocity, 
                tugboat_dim, 
                max_rope_length,
                linear_drag_coeff=1.0,    # Translational drag coefficient
                angular_drag_coeff=3.0
                ):
        """
        Observation Space:
        Type: Box(11)

        Action Space:
        Type: Box(5)

        """
        self.grid_size = grid_size
        self.dock_position = dock_position
        self.target_position = target_position
        self.dt = frame_update

        self.ship_dim = ship_dim # [length, breadth]
        self.ship_mass = ship_mass
        self.ship_inertia = ship_inertia
        self.ship_velocity = ship_velocity
        self.angular_velocity = ship_angular_velocity

        self.tugboat_dim = tugboat_dim # [length, breadth]
        self.max_rope_length = max_rope_length
        self.front_offset = ship_dim[0]
        self.z = np.sqrt((self.ship_dim[0]**2) + (self.ship_dim[1]**2))

        # [xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l]
        #TODO: Fix low and high values
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -np.pi/2, 0.0, 0.0, -np.pi/2, 0.0, 0.0]),
                                            high=np.array([100.0, 100.0, 2 * np.pi, 100.0, 100.0, np.pi/2, 100.0, 100.0, np.pi/2, 100.0, 100.0]),
                                            dtype=np.float32)
        
        #TODO: first try with action space consisting of x, y velocity components of tugboats
        # [Ft1, Ft2, alphat1, alphat2, taus]
        # self.action_space = spaces.Box(low=np.array([0.0, 0.0, -np.pi, -np.pi, 0.0]),
        #                                high=np.array([100.0, 100.0, np.pi, np.pi, 10.0]),
        #                                dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0]),
                                       high=np.array([10.0, 10.0, 10.0, 10.0]),
                                       dtype=np.float32)

        self.obstacles = [
            (0, 150, 200, 50)
        ] # (x, y, width, height)

        self.linear_drag_coeff = linear_drag_coeff
        self.angular_drag_coeff = angular_drag_coeff
        
        self.reset()


    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]


    def reset(self):
        # Set tugboats at front-left and front-right of the ship
        # Calculate positions based on ship's dimensions
        front_offset = self.ship_dim[0]

        # Tugboat initial positions adjusted for front-left and front-right
        self.state = np.array([
            50.0, 50.0, 0.0,  # Ship initial position (xs, ys, thetas)
            55.0 + front_offset, 50.0, 0.0,  # Tugboat 1 at front-right (xt1, yt1, thetat1)
            55.0 + front_offset, 50.0+self.ship_dim[1], 0.0,  # Tugboat 2 at front-left (xt2, yt2, thetat2)
            50.0,  # Distance to the target (ds)
            3.0  # Rope length
        ])
        return self.state



    def check_collision(self, x, y, length, breadth, object_type='ship'):
        """
        Enhanced collision check that prevents collisions between ship, tugboats
        and obstacles.
        """
        # Check obstacle collisions
        for (ox, oy, owidth, oheight) in self.obstacles:
            if (x < ox + owidth and x + length > ox and
                y < oy + oheight and y + breadth > oy):
                return True
        
        # Check ship-tugboat collisions
        if object_type != 'ship':
            ship_x, ship_y = self.state[0], self.state[1]
            if (x < ship_x + self.ship_dim[0]/2 and x + length > ship_x - self.ship_dim[0]/2 and
                y < ship_y + self.ship_dim[1]/2 and y + breadth > ship_y - self.ship_dim[1]/2):
                return True
        
        return False


    def step(self, action):
        vx1, vy1, vx2, vy2 = action

        # Unpack the state
        xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l = self.state


        # Compute the single attachment point at the center front of the ship
        P_fr = np.array([xs + self.front_offset * np.cos(thetas),
                        ys + self.front_offset * np.sin(thetas)])  # front right of ship
        P_fl = np.array([xs + self.z*np.cos(thetas+np.arctan(self.ship_dim[1]/self.ship_dim[0])),
                         ys + self.z*np.sin(thetas+np.arctan(self.ship_dim[1]/self.ship_dim[0]))])

        # Predict new tugboat positions
        new_xt1 = xt1 + vx1 * self.dt
        new_yt1 = yt1 + vy1 * self.dt
        new_xt2 = xt2 + vx2 * self.dt
        new_yt2 = yt2 + vy2 * self.dt

        # Check for potential collisions before updating positions
        # if self.check_collision(new_xt1 - self.tugboat_dim[0] / 2, 
        #                         new_yt1 - self.tugboat_dim[1] / 2, 
        #                         *self.tugboat_dim, 'tugboat'):
        #     vx1, vy1 = 0, 0  # Stop tugboat 1 if collision detected

        # if self.check_collision(new_xt2 - self.tugboat_dim[0] / 2, 
        #                         new_yt2 - self.tugboat_dim[1] / 2, 
        #                         *self.tugboat_dim, 'tugboat'):
        #     vx2, vy2 = 0, 0  # Stop tugboat 2 if collision detected

        # Update tugboat positions
        xt1 += vx1 * self.dt
        yt1 += vy1 * self.dt
        xt2 += vx2 * self.dt
        yt2 += vy2 * self.dt

        # Enforce constant rope length and connection to the single front attachment point
        rope1_vector = np.array([xt1, yt1]) - P_fr
        rope2_vector = np.array([xt2, yt2]) - P_fl

        # Fix the lengths to the desired max_rope_length
        rope1_length = np.linalg.norm(rope1_vector)
        rope2_length = np.linalg.norm(rope2_vector)

        if rope1_length > self.max_rope_length:
            xt1, yt1 = P_fr + (rope1_vector / rope1_length) * self.max_rope_length

        if rope2_length > self.max_rope_length:
            xt2, yt2 = P_fl + (rope2_vector / rope2_length) * self.max_rope_length

        # Update the tugboat positions to ensure constant rope lengths
        xt1, yt1 = P_fr + rope1_vector
        xt2, yt2 = P_fl + rope2_vector

        # Compute the forces exerted by the ropes based on their new fixed positions
        force_front_1 = (rope1_vector / self.max_rope_length) * 2.0
        force_front_2 = (rope2_vector / self.max_rope_length) * 2.0

        # Calculate net force on the ship (translation)
        net_force = force_front_1 + force_front_2

        # Add linear drag force to reduce velocity
        linear_drag_force = -self.linear_drag_coeff * self.ship_velocity
        net_force += linear_drag_force

        # Calculate acceleration of the ship from the net force
        acceleration = net_force / self.ship_mass  # a = F / m

        # Update the ship's velocity based on acceleration
        self.ship_velocity += acceleration * self.dt  # v = v + a * dt

        # Update ship's position based on the new velocity
        xs += self.ship_velocity[0] * self.dt
        ys += self.ship_velocity[1] * self.dt

        # Calculate torque on the ship (rotation)
        # Only one attachment point is used, so the torque is simpler
        torque = (np.cross(P_fr - np.array([xs, ys]), force_front_1 + force_front_2))

        # Add angular drag torque to reduce rotational velocity
        angular_drag_torque = -self.angular_drag_coeff * self.angular_velocity
        torque += angular_drag_torque

        # Calculate angular acceleration from torque
        angular_acceleration = torque / self.ship_inertia  # alpha = τ / I

        # Update angular velocity
        self.angular_velocity += angular_acceleration * self.dt  # ω = ω + α * dt

        # Update ship's orientation based on the angular velocity
        thetas += self.angular_velocity * self.dt

        # Update the distance to the dock
        new_ds = np.linalg.norm(np.array([xs, ys]) - np.array([250, 0])) 

        # Collision check
        collision = False
        # Adjust reward
        reward = -1  # Default small penalty
        if collision:
            reward -= 50  # High penalty for hitting an obstacle
        if new_ds <= 0.5 and abs(thetas) < 0.1:
            reward += 100  # High positive reward if docking is successful

        # End the episode if docked correctly
        done = new_ds <= 0.5

        # Update the state
        self.state = np.array([xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, new_ds, l])
        return np.array(self.state, dtype=np.float32), reward, done, {}



    def render(self):
        # Create a figure and axis
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()
        else:
            self.ax.clear()

        self.ax.set_facecolor('lightblue')

        # Set grid limits
        self.ax.set_xlim(0, 500)
        self.ax.set_ylim(0, 500)
        self.ax.set_aspect('equal')
        self.ax.set_xticks(np.arange(0, 501, 50))  # 10 unit spacing for X-axis
        self.ax.set_yticks(np.arange(0, 501, 50))  # 10 unit spacing for Y-axis
        self.ax.grid(which='both', linestyle='-', linewidth=0.5, color='gray')

        # Draw the dock
        # dock_x, dock_y = self.dock_position
        dock_patch = patches.Rectangle(xy=self.dock_position, 
                                    width=100, 
                                    height=30, 
                                    edgecolor='brown', 
                                    facecolor='sienna')
        self.ax.add_patch(dock_patch)

        # Unpack state
        xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l = self.state

        # Define the ship as a rectangle
        ship_length, ship_breadth = self.ship_dim
        ship_patch = patches.Rectangle(
            (xs, ys), ship_length, ship_breadth,
            angle=np.degrees(thetas), edgecolor='blue', facecolor='lightblue', lw=2)

        # Add the ship to the plot
        self.ax.add_patch(ship_patch)

        # Define tugboats as smaller rectangles
        tugboat_length, tugboat_breadth = self.tugboat_dim

        # Tugboat 1
        tugboat1_patch = patches.Rectangle(
            (xt1, yt1), tugboat_length, tugboat_breadth,
            angle=np.degrees(thetat1), edgecolor='green', facecolor='lightgreen', lw=2)

        # Tugboat 2
        tugboat2_patch = patches.Rectangle(
            (xt2, yt2), tugboat_length, tugboat_breadth,
            angle=np.degrees(thetat2), edgecolor='orange', facecolor='lightyellow', lw=2)

        # Add tugboats to the plot
        self.ax.add_patch(tugboat1_patch)
        self.ax.add_patch(tugboat2_patch)

        # Compute attachment points in the global frame
        P_fr = np.array([xs + self.front_offset * np.cos(thetas),
                    ys + self.front_offset * np.sin(thetas)])
        P_fl = np.array([xs + self.z*np.cos(thetas+np.arctan(self.ship_dim[1]/self.ship_dim[0])),
                         ys + self.z*np.sin(thetas+np.arctan(self.ship_dim[1]/self.ship_dim[0]))])

        # Draw ropes as lines
        self.ax.plot([P_fr[0], xt1], [P_fr[1], yt1], '-', lw=1.5, color='black')  # Rope 1 (front)
        self.ax.plot([P_fl[0], xt2], [P_fl[1], yt2], '-', lw=1.5, color='black')  # Rope 2 (back)

        # Plotting obstacles
        # for (ox, oy, owidth, oheight) in self.obstacles:
        #     obs_patch = patches.Rectangle((ox, oy), width=owidth, height=oheight, edgecolor='red', facecolor='lightcoral')
        #     self.ax.add_patch(obs_patch)

        # Labels and Legend
        self.ax.set_title('Ship Towing Environment')
        self.ax.legend(loc='upper right')
        self.ax.grid(True)

        # Display the plot
        plt.pause(0.001)  # Small pause to update the plot
        plt.draw()



#env variables
grid_size = 500
dock_position = (200, 450)
target_position = (250, 400)
frame_update = 0.01

#ship variables
ship_dim = (50,8) # (length, breadth)
ship_mass = 5.0
ship_inertia = 0.1
ship_velocity = 0.025
ship_angular_velocity = 0.0000000001


#tugboat variables
tugboat_dim = (5,2) # (length, breadth)
max_rope_length = 10.0



env = ShipTowEnv(grid_size=grid_size,
                dock_position=dock_position, 
                target_position=target_position,
                frame_update=frame_update,
                ship_dim=ship_dim, 
                ship_mass=ship_mass, 
                ship_inertia=ship_inertia,
                ship_velocity=ship_velocity,
                ship_angular_velocity=ship_angular_velocity, 
                tugboat_dim=tugboat_dim, 
                max_rope_length=max_rope_length
                )

state = env.reset()
done = False

while not done:
    action = env.action_space.sample() 
    state, reward, done, _ = env.step(action)
    xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l = state
    if xs > target_position[0]:
        done=True
    env.render()

env.close()