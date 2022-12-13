### Setup

#### 1) Download simulator from the link below:

[Download](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

#### 2) Install Python 3.8

#### 3) Install packages in requirement.txt

#### 4) Run the simulator and put it on autonomous mode - Make sure the firewall is off

#### 5) Set the configuration file (drive-dev.config)
##### set active to 0 to disable attack, set to 1 to active attack like below
    active = 1 
##### set type to "random", "turn_left" or "turn_right" like below
    type = turn_right 

#### 6) Run the command below in project root directory 
```
python3 app.py ACTIVE_PROFILE=dev
```

Once the application starts the car starts moving and based on the configuration in config file it acts.

#### 7) The generated result is getting store in record.csv file
#### 8) During the execution, it is not possible to change the configuration. You should stop and start the app.

