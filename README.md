# Robot Robbers
Develop a system for playing robot robbers.
The Robot Robbers game is an interactive 2D game where you, the player, control 5 robots trying to steal money from angry scrooges.

The objective  is to reach the highest possible reward in 2 wall-clock minutes. Balance the trade off between time and performance well!

## About the game
Every game runs in a 128x128 grid environment. Every game is initialized with:

1. 5 robots (controlled by the player)
2. 7 scrooges (controlled by the game)
3. 5 cashbags
4. 3 dropspots
5. Between 2-5 obstacles, the height and width of which range between 1 and 20.

All of these are randomly placed at every game start.
You will recive the states at your prediction endpoint for every game tick.

### Controlling robots
You have to control all 5 robots for every game tick. Every robot kan only move 1 step at a time in either horizontal, vertical or diagonal direction. Movement instructions has to be provided as delta `x`and `y`.

For example:
```python
moves = [
  1, 1,    # Move robot 0 one cell to the right, one cell down
  -1, -1,  # Move robot 1 one cell to the left, one cell up
  0, 0     # Make robot 2 stand still
]
```

### Cashbags & Dropspots
When a robot robber intersects with a cashbag, the robot picks it up. When a robot carrying cashbags intersects with a dropspot, the cashbags are deposited and a reward is provided.

The reward of depositing cashbags increases exponentially by the number of cashbags carried, e.g.:

1. Carrying 1 cashbag -> reward of 1
2. Carrying 2 cashbags -> reward of 4
3. Carrying 3 cashbags -> reward of 9

However, robots become burdened by carrying cashbags and they move slower the more they carry:
1. Robot speed (0 cashbags): 1 ticks / move
2. Robot speed (1 cashbag): 2 ticks / move
3. Robot speed (2 cashbags): 3 ticks / move

The number of cashbags on the screen always remains the same. Cashbacks respawn when they are deposited or when they are taken away by scrooges.


### Scrooges

The scrooges are the game antagonists. They will try their very best to keep the cashbags from being stolen.

Initially, scrooges will move around randomly on the map.

If a robot carrying cashbags intersects with a scrooge, the cashbags are taken away and the player receives a -3 reward penalty.

If a robot is within a distance of 15 units of a scrooge, the scrooge will chase the robot until:

* The scrooge reaches the robot, at which point the robot will not be chased again by any scrooge for 100 game ticks.
* The robot comes out of range, at which point the scrooge will wander randomly again.

Scrooges always move at the speed of 2 ticks / move.

## Rules
* Robots and scrooges cannot move outside of the grid.
* Robots and scrooges can only move one unit in either direction in a single game tick.
* Robots and scrooges cannot move through obstacles.

## Interaction
You'll recive a `RobotRobbersPredictRequestDto` which contain the following:
```python
class RobotRobbersPredictRequestDto(BaseModel):
    state: List[List[List[int]]]
    reward: float
    is_terminal: bool
    total_reward: float
    game_ticks: int
```
Where **state** is composed of the following:
Given an observation matrix $M \in \mathbb{Z}^{6 \times 10 \times 4}$, the contents are as follows:

1. $M_{0}$ is an array of 4-d vectors containing the $x, y, w, h$ of all **robot robbers** ($w, h$ is always $1$).
2. $M_{1}$ is an array of 4-d vectors containing the $x, y, w, h$ of all **scrooges** ($w, h$ is always $1$).
3. $M_{2}$ is an array of 4-d vectors containing the $x, y, w, h$ of all **cashbags** ($w, h$ is always $1$).
4. $M_{3}$ is an array of 4-d vectors containing the $x, y, w, h$ of all **dropspots** ($w, h$ is always $1$).
5. $M_{4}$ is an array of 4-d vectors containing the $x, y, w, h$ of all **obstacles**.
6. $M_{5}$ is an array of 4-d vectors where the first element of the vector is the number of cashbags carried by the robot robber with the same index. The rest of the vector elements are always $0$.
   - For example, given the robber $i$ at $M_{0,i}$, the number of cashbags carried by this robber is $M_{5, i, 0}$.

Each row of the observation matrix contains 10 4-d vectors, but not all vectors represent active game elements.
Inactive game elements (e.g., the last 5 vectors in $M_{0}$) are represented by their position being placed outside the game grid `(-1, -1)`. 
> For example, the vector of an inactive cashbag (cashbags are inactive while being carried by a robot) will always be `(-1, -1, 1, 1)`.

* **Reward** is when you gain points, this will be appearent in the reward.
* **total_reward** is the total score for your current game. 
* **game_ticks** is the number of game tick currently running.

From these information you should be able to predict your next move and return:
```python
class RobotRobbersPredictResponseDto(BaseModel):
    moves: List[int]
```
which is a 10-d vector of moves: <br>
moves = [ $Δ_{x,0}$, $Δ_{y,0}$, $Δ_{x,1}$, $Δ_{y,2}$, ... $Δ_{x,4}$, $Δ_{y,4}$ ] <br>
where, $Δ_{x,n}$ and $Δ_{y,n}$ are the change in x and y direction for robot n. 


### Install dependencies
```shell 
pip install -r requirements.txt
```
