# :round_pushpin::arrow_double_up::checkered_flag:Wall Jump

Followed: [An Introduction to Unity ML-Agents](https://towardsdatascience.com/an-introduction-to-unity-ml-agents-6238452fcf4c)

## The Wall Jump Environment

3 situations:

* **No wall**. The agent simply needs to go to the green tile.
* **Small wall**. The agent needs to learn to jump in order to reach the green tile.
* **Big wall**. The agent will not be able to jump as high as the wall is so it needs to push the white block in order to jump on it to be able to jump over the wall.

## Policies

* **SmallWallJump**. Learned during the no wall and low wall situations.
* **BigWallJump**. Learned during the high wall situations.

## Reward system

* **-0.005** for every step.
* **+1** if the agent reaches the green tile.
* **-0.01** if the agent falls.

Goal: mean reward of 0.8

## Action space

* **Forward Motion**: UP / DOWN / NO ACTION
* **Rotation**: ROTATE LEFT / ROTATE RIGHT / NO ACTION
* **Side Motion**: LEFT / RIGHT / NO ACTION
* **Jump**: JUMP / NO ACTION

## Training

1. Remove the brains from the agent.
2. Modify the total training steps for the two policies.
3. Set the training parameters in the WallJump.yaml file stored at config/ppo.
4. Start training (command line)

```command line
mlagents-learn config/ppo/WallJump.yaml --run-id=”WallJump_FirstTrain”
```

5. Run the Unity scene by  pressing the ▶️ button at the top of the Editor.
6. Observe training with tensorboard

```command line
tensorboard --logdir results --port 6006
```

7. When the training is finished move the saved model files contained in /results to (Unity project folder)/Assets/ML-Agents/Examples/WallJump/TFModels.
8. Drag the .nn files to the corresponding Placeholders:
   * In Agent Behavior Parameters, SmallWallJump.nn to Model Placeholder.
   * SmallWallJump.nn file to No Wall Brain Placeholder.
   * SmallWallJump.nn file to Small Wall Brain Placeholder.
   * BigWallJump.nn file to No Wall Brain Placeholder.
