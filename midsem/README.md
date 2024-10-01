# MID SEM:

NAME: Mohammad Saifullah Khan  
ROLL NO.: 21169  
DEPARTMENT: EECS

## Run the Code:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install gymnasium numpy matplotlib
python3 q_with_epsilon.py
```
Results are saved in home directory.

## ANSWER:

Q-learning without epsilon-greedy and with epsilon-greedy is used.  
Epsilon-greedy is implemented as decaying to balance exploration and exploitation.  
At the early stages of training, agent should explore more to gather information about rewards associated with states and actions.   
As agent learns over time it should exploit the its learned knowledge.  

High epsilon -> agent takes random actions more frequently.  
Low epsilon -> agent chooses action with maximum q-value.  

In the code,  
```epsilon_start``` -> epsilon at the start of training process.   
```epsilon_end``` -> minimum value of epsilon decay.  
```epsilon_decay_steps``` -> the rate at which the epsilon value decreases.  


To run without epsilon-greedy, change the following variables inside ```main()``` function.  
```bash
epsilon_start = 0
epsilon_end = 0
```

To run with with epsilon-greedy, change the following variables inside ```main()``` function.  
```bash
epsilon_start = <some value>
epsilon_end = <some value>
```

## About Code:
```ModeTSP``` class initialize the TSP environment.  

```QLearning``` class initializes agent for q-learning for the tsp environment.  
Hyperparameter:

| Hyperparameter | Value |
| -------------- | ----- |
| Learning Rate (```alpha```) | 0.03 |
| Discount Factor (```gamma```) | 0.99 |
| Epsilon Start (```epsilon_start```) | 0.5 |
| Minimum epsilon (```epsilon_end```) | 0.01 |
| Epsilon Decay Rate (```epsilon_decay_steps```) | 300 |


Initializations:  
```self.Q_table```: dictionary to store Q-values,  
```self.td_errors```: list to track td errors,  
```self.epsilon_decay```: rate at which epsilon decreases to shift from exploration to exploitation,  




## Results:
