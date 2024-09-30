# MID SEM:

NAME: Mohammad Saifullah Khan  
ROLL NO.: 21169  
DEPARTMENT: EECS


## ANSWER:

Q-learning without epsilon greedy and with epsilon greedy is used.  
Epsilon-greedy is implemented as decaying to balance exploration and exploitation.  
At the early stages of training, agent should explore more to gather information about rewards associated with states and actions.   
As agent learns over time it should exploit the its learned knowledge.  

High epsilon -> agent takes random actions more frequently.  
Low epsilon -> agent chooses action with maximum q-value. 

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
