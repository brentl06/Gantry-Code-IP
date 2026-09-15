# Code Additions/Changes
## `data_collector_new.py`
Code tied to the mocap system was removed, as motion capture is only used for the stationary system, not the gantry system.
Additionally, lines 883-897 of `main()` will construct the controller that connects to the gantry and either move or stand still, based on user input.
