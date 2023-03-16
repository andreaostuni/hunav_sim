# Hunav Evaluator

A ROS2 python package to compute different human-aware navigation metrics about the simulations performed with the HuNav Simulator.

**Tested in ROS2 Foxy** 


## Dependencies

* numpy
* hunav_msgs
* hunav_agent_manager


## Description

This package subscribes to a set of topic published by the Hunav Agent Manager:

* ```/human_states``` this topic publishes messages of type *hunav_msgs/msg/Agents* with the status of the human agents for each time step.
* ```/robot_states``` this topic publishes messages of type *hunav_msgs/msg/Agent* with the status of the robot for each time step.
    
This information is stored and then is employed to compute a set of metrics related to the social robot navigation. 

The recording process is automatic. 

It starts when the first message in the subscribed topics is received.
The recording stops when a certain time passes without receiving data through the topics.

Once the recording has stopped, the computation of the metrics is performed and an output txt file is generated.


## Configuration and Parameters

The user must specify the set of metrics to be computed and some parameters about the metrics generation.
This values must be indicated in the yaml file metrics.yaml, placed in the config directory. After compilation this file is moved to the install directory of the ROS workspace. 

### Global parameters

* ```frequency```. Indicate the frequency of capturing the data (it must be slower than data publishing). If the value is set to zero, the data is captured at the same frequency than it is published.
* ```experiment_tag```. A string that can be used to identify different scenarios or experiments. This tag will be stored in the output results file. 
* ```result_file```. Full path and name of the output text file. If a new experiment is performed and the file already exists, the evaluator wil open the file and will add the computed metrics in a new row at the end of the file. This allows to repeat experiments and to store the results as rows in the same file for easier post-processing.
* ```metrics```. List with the names of the metrics to be computed. The user can just comment/uncomment the desired metrics.   

### Metrics configuration

Currently, the metrics implemented are those employed in our previous work [1] and the SEAN simulator [2]. The metrics from the SocNavBench [3], Crowdbot [4] and the compilation indicated in the work of Gao et al. [5], are being studied and included.

The user can configure the metrics to be computed in two different ways:

- By editing the metrics yaml file, and changing the boolean parameters of the metrics (true to compute the metric, false to ignore it).
- By using a RViz2 panel to select/unselect the metrics and to store them. See the [hunav_rviz2_panel](https://github.com/robotics-upo/hunav_sim/tree/master/hunav_rviz2_panel) for more information on how to use the panel.  
 
Example snippet of the metrics configuration file:

```yaml
    metrics:
    
      # Pérez-Higueras, N., Caballero, F. & Merino, L. 
      # Teaching Robot Navigation Behaviors to Optimal RRT Planners. 
      # Int.J. of Soc. Robotics 10, 235–249 (2018). https://doi.org/10.1007/s12369-017-0448-1
      time_to_reach_goal: true
      path_length: false
      cumulative_heading_changes: true
      avg_distance_to_closest_person: true
      minimum_distance_to_people: true
      intimate_space_intrusions: true
      personal_space_intrusions: true
      social_space_intrusions: true
      group_intimate_space_intrusions: true
      group_personal_space_intrusions: true
      group_social_space_intrusions: true
      
      
      # N. Tsoi et al., "SEAN 2.0: Formalizing and Generating Social Situations
      # for Robot Navigation," in IEEE Robotics and Automation Letters, vol. 7,
      # no. 4, pp. 11047-11054, Oct. 2022, doi: 10.1109/LRA.2022.3196783.
      completed: true
      minimum_distance_to_target: true
      final_distance_to_target: true
      robot_on_person_collision: false
      person_on_robot_collision: false
      time_not_moving: true
      
      # SocNavBench: A Grounded Simulation Testing Framework for Evaluating Social Navigation
      #ABHIJAT BISWAS, ALLAN WANG, GUSTAVO SILVERA, AARON STEINFELD, and HENNY AD-MONI, Carnegie Mellon University
      avg_speed: false
      avg_acceleration: false
      avg_overacceleration: false
      
      # Learning a Group-Aware Policy for Robot Navigation
      # Kapil Katyal ∗1,2 , Yuxiang Gao ∗2 , Jared Markowitz 1 , Sara Pohland 3 , Corban Rivera 1 , I-Jeng Wang 1 , Chien-Ming Huang 2 
      avg_pedestrian_velocity: true

      # metrics based on Social Force Model employed in different papers
      # See the compilation of Gao et al. [5]
      social_force_on_agents: true
      social_force_on_robot: true
      # social work function employed in this planner:
      # https://github.com/robotics-upo/social_force_window_planner
      social_work: true
      obstacle_force_on_robot: true
      obstacle_force_on_agents: true
```

### Metrics description

Currently, the metrics implemented are:

Metrics from our previous work [1]:

- *Time to reach the goal* $(T_p)$. Time $(seconds)$ since the robot start the
navigation until the goal is correctly reached.

$$ T_p = (T_{goal} - T_{ini}) $$

- *Path length* $(L_p)$. The length of the path $(meters)$ followed by the robot from the initial point to the goal position.

$$ L_p = \sum_{i=1}^{N-1}||x_r^i - x_r^{i+1}||_2$$

- *Cumulative heading changes*, $(CHC)$. It counts the cumulative heading changes $(radians)$ of in the robot trajectory measured by angles between successive waypoints. It gives a simple way to check on smoothness of path and energy so a low value is desirable.

$$ CHC = \sum_{i=1}^{N-1}(h_r^i - h_r^{i+1})$$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;where $h_r^i$ indicates the heading of the robot in the position i. The angles and their difference are normalized between $-\pi$ and $\pi$ .

- *Average distance to closest person* $(CP_{avg})$. A measure
of the mean distance $(meters)$ from the robot to the closest person
along the trajectory.

$$ CP_{avg} = \frac{1}{N} \sum_{i=1}^{N}(||x_r^i - x_{cp}^i||_2)$$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;where $x_{cp}^i$ indicates the position of the closest person to the robot at step i.

- *Minimum and maximum distance to people* ( $CP_{min}$ and $CP_{max}$ respectively). The values of the minimum and the maximum distances $(meters)$ from the robot to the people along the trajectory. It can give an idea of the dimension of the robot trajectory with respect to the people in the space.

$$ CP_{min} = min \{||x_r^i-x_{cp}^i||_2  \forall i \in N \} $$

$$ CP_{max} = max \{||x_r^i-x_{cp}^i||_2  \forall i \in N \} $$

- *Intimate/Personal/Social space intrusions* $(CP_{prox})$. This metric is based on the Proxemics theory which define personal spaces around people for interaction. These areas are defined as:

    - Intimate. Distance shorter than 0.45m.
    - Personal. Distance between 0.45 and 1.2m.
    - Social. Distance between 1.2 and 3.6m.
    - Public. Distance longer than 3.6m.

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Thus, the metric classifies the distance between the robot and the closest person at each time step in one of the Proxemics spaces and obtain a percentage of the time spent in each space for the whole trajectory:

$$ CP_{prox}^k = (\frac{1}{N} \sum_{j=1}^{N} F(||x_r^j - x_{cp}^j||_2 \lt \delta^k)) * 100$$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; where N is the total number of time steps in the trajectory, $\delta$ defines the distance range for 
classification defined by k = {Intimate, Personal, Social + Public}, and F(·) is the indicator function.

- *Interaction space intrusions* $(IS_{prox})$. This metric is inspired by the work of Okal and Arras [6] in formalizing social normative robot behavior, and it is related to groups of interacting persons. It measures the percentage of time spent by the robot in the three Proxemics spaces considered with respect to an interaction area formed by a group of people that are interacting with each other. The detection of the interaction area of the group is based on the detection of F-formations. A F-formation arises whenever two or more people sustain a spatial and orientational relationship in which the space between them is
one to which they have equal, direct, and exclusive access

$$ IS_{prox}^k = (\frac{1}{N} \sum_{j=1}^{N} F(||x_r^j - x_f^j||_2 \lt \delta^k)) * 100 $$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;where $x_f^j$ determines the center of the closest formation or group of people $f$ to the robot at step $j$.


Metrics from the SEAN simulator [2]:

- *Completed* $C$. true when the robot's final pose is within a specified threshold distance of the goal.
  
$$ C = F(\({||x_r^N - x_{g}||}_{2}  - \gamma_g \) \lt 0.0) $$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; where $x_r^N$ is the robot pose in the last time step of the trajectory, $x_{g}$ is the goal position and  $\gamma_{g}$ is the goal radius. $F(·)$ is the indicator function.

- *Total Path Length*. This metric is equivalent *Path Length* metric of [1].

- *Minimum Distance to Target*, $TD_{min}$ $(meters)$. The minimum distance to the target position reached by the robot along its path.

$$ TD_{min} = min \{||x_r^i-x_{g}||_2 \forall i \in N \} $$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; where $x_r^i$ is the robot pose in the time step $i$ of the trajectory, and $x_{g}$ is the target position.

- *Final Distance to Target* $(meters)$. Distance between the last robot position and the target position.

$$ TD_{final} = ||x_r^N -x_{g}||_2 $$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; where $x_r^N$ is the robot pose in the last time step of the trajectory.

- *Time Not Moving*. Seconds that the robot was not moving. We consider that the robot is not moving when the linear velolcity is lower than $0.001$ $m/s$ and the angular velocity lower than $0.01$ $rad/s$.


- *Robot on Person Collision*, $RPC$, and *Person on Robot Collision*, $PRC$. Number of times robot collides with a person and vice versa. The SEAN simulator employs the Unity colliding tools to detect the collisions. However, we want our metrics to be independent of any simulator. Therefore, this metric has been implemented by using the positions, the orientations and the velocities of the robot and agents to determine who is running into the other. The collisions are primarily computed as:

$$ Collisions = \sum_{i=1}^{N} F(\({||x_r^i - x_{cp}^i||}_{2}  - \gamma_{r} - \gamma_{cp}\) \lt 0.01) $$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; where $x_r^i$ is the robot pose in the time step $i$ of the trajectory, $x_{cp}^i$ is the pose of the closest person to the robot at time step $i$. $\gamma_{r}$ is the radius of the robot and $\gamma_{cp}$ is the radius of the closest person. Then, we check the relative orientations between the agent and the robot and their velocities to determine who is running into the other.  

Metrics based on Social Force Model [7,8,9]. The mathematical equations of the computation of the forces can be consulted [here](https://github.com/robotics-upo/lightsfm):

- *Social Force on agents*. Summatory of the modulus of the social force provoked by the robot in the agents  

- *social_force_on_robot*. Summatory of the modulus of the social force provoked by the human agents in the robot.
  
- *social work*. It is based on the concept of the "**social work**" performed by the robot ($W_{r}$), and the social work provoked by the robot in the surrounding pedestrians ($W_{p}$)

$$ W_{social} = W_{r} + \sum W_{p_{i}} $$

With:

  - $W_{r}$  The summatory of the modulus of the robot social force and the robot obstacle force along the trajectory. According to the SFM.
  - $W_{p}$  The summatory of the modulus of the social forces provoked by the robot for each close pedestrian along the trajectory. According to the SFM. The social work function is employed in this [local planner](https://github.com/robotics-upo/social_force_window_planner)

- *obstacle_force_on_agents*. Summatory of the modulus of the obstacle force provoked by the obstacles in the agents (no other agents or robot are considered). 
  
- *obstacle_force_on_robot*. Summatory of the modulus of the obstacle force provoked by the obstacles in the robot (the human agents are not considered).
 

<!-- - *Static Obstacle Collision*: number of times the robot collides with a static obstacle. -->

- *Robot on Person Personal Distance Violation*, *Person on Robot Personal Distance Violation*, *Intimate Distance Violation* and *Person on Robot Intimate Distance Violation*. These metrics are not used. We are employing the similar *Intimate/Personal/Social space instrusions* [1] instead. Instead of giving the number of times the robot approaches a person within the personal distance, a percentage of the total time is given.

- *Path Irregularity* $(radians)$ [NOT IMPLEMENTED]. Total rotations in the robot's traveled path greater than the total rotations in the search-based path from the starting pose. We have not implemented this metric for the moment. We do not use the path of the global planner as input of the metric functions, as well as not all the navigation systems are using an unique global path without replanning. Anyway, we are considering to add the global path information into the metric function inputs.

- *Path Efficiency* $(meters)$ [NOT IMPLEMENTED]. Ratio between robot's traveled path and geodesic distance of the search-based path from the starting pose. We have not implemented this metric for the moment. We do not use the path of the global planner as input of the metric functions, as well as not all the navigation systems are using an unique global path without replanning. Anyway, we are considering to add the global path information into the metric function inputs.




## Adding new metrics

The user can easily extend the current set of available metrics.

To do so, a new metric function must be implemented in the ```hunav_metrics.py``` Python file. 
Every metric function must take the following input parameters:

* ```robot``` a list of messages *hunav_msgs/msg/Agent* where all information recorded at each time step about the social robot is stored.
* ```agents``` a list of messages *hunav_msgs/msg/Agents* where all information recorded about the human agents are stored.

See the [hunav_msgs](https://github.com/robotics-upo/hunav_sim/tree/master/hunav_msgs) package to obtain more information about the data stored for each agent. 

After programming the new metric function, you must associate a name to it in the *metrics* list at the end of the ```hunav_metrics.py``` file. 

After compiling, the user can add the new function name to the ```metrics.yaml``` file.



## TODO:

- [ ] Program a set of ROS2 services to start/stop the data recording.
- [X] Include some related to the forces of the SFM.
- [ ] Complete the set of metrics

## References:

[1] N. Perez-Higueras, F. Caballero, and L. Merino, "Teaching Robot Navigation Behaviors to Optimal RRT Planners," International Journal of Social Robotics, vol. 10, no. 2, pp. 235–249, 2018.

[2] N. Tsoi, A. Xiang, P. Yu, S. S. Sohn, G. Schwartz, S. Ramesh, M. Hussein, A. W. Gupta, M. Kapadia, and M. Vázquez, "Sean 2.0: Formalizing and generating social situations for robot navigation," IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 11 047–11 054, 2022.

[3] A. Biswas, A. Wang, G. Silvera, A. Steinfeld, and H. Admoni,"Socnavbench: A grounded simulation testing framework for evaluating social navigation," ACM Transactions on Human-Robot Interaction, jul 2022. [Online](https://doi.org/10.1145/3476413).

[4] F. Grzeskowiak, D. Gonon, D. Dugas, D. Paez-Granados, J. J. Chung, J. Nieto, R. Siegwart, A. Billard, M. Babel, and J. Pettré, “Crowd against the machine: A simulation-based benchmark tool to evaluate and compare robot capabilities to navigate a human crowd,” in 2021 IEEE International Conference on Robotics and Automation (ICRA), 2021, pp. 3879–3885. 

[5] Y. Gao and C.-M. Huang, "Evaluation of socially-aware robot navigation," Frontiers in Robotics and AI, vol. 8, 2022.[Online](https://www.frontiersin.org/articles/10.3389/frobt.2021.721317)

[6] Okal B, Arras KO, "Formalizing normative robot behavior," In: Social robotics: 8th international conference, ICSR 2016,Kansas City, MO, USA, November 1-3, 2016 Proceedings.Springer International Publishing, pp 62–71. [Online](https://doi.org/10.1007/978-3-319-47437-3_7)

[7] Helbing, Dirk & Molnar, Peter. (1998). "Social Force Model for Pedestrian Dynamics". Physical Review E. 51. 10.1103/PhysRevE.51.4282. 

[8] Moussaid M, Helbing D, Garnier S, Johansson A, Combe M, et al. (2009) "Experimental study of the behavioural mechanisms underlying self-organization in human crowds". Proceedings of the Royal Society B: Biological Sciences 276: 2755–2762.

[9] Moussaïd, Mehdi & Perozo, Niriaska & Garnier, Simon & Helbing, Dirk & Theraulaz, Guy. (2010). "The Walking Behaviour of Pedestrian Social Groups and Its Impact on Crowd Dynamics". PloS one. 5. e10047. 10.1371/journal.pone.0010047.


