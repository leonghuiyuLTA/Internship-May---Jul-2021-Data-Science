The current position-time, velocity-time and acceleration-time graphs can be seen in the respective pngs.
To observe the animation, run the "Trained Agent.py" file.

"Trainer.py" is the code that runs the trainer. The functions are:
1. get_state(), which classifies the current state of the bus into the tiles
2. choose_action(), which decides which action the agent will take

"autobus_env.py" is the environment. The function get_reward() shows how the rewards are calculated.

"Rendering.py" is the code which shows the animation.

The agent ends his journey at 228m, and exceeds the speed limit at some point.
The score at the beginning of training is insanely negative, but reaches an acceptable position nearer the end.
