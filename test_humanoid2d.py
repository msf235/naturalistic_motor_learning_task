import humanoid2d as h2d

# Create a Humanoid2dEnv object
env = h2d.Humanoid2dEnv(render_mode='human')
env.reset()
zac = 9*[0]
for k in range(200):
    env.step(zac)
