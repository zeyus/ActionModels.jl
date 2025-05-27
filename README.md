```@meta
EditURL = "docs/julia_files/E_others/README.jl"
```

# ActionModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ilabcode.github.io/ActionModels.jl)
[![Build Status](https://github.com/ilabcode/ActionModels.jl/actions/workflows/CI_full.yml/badge.svg?branch=main)](https://github.com/ilabcode/ActionModels.jl/actions/workflows/CI_full.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ilabcode/ActionModels.jl/branch/main/graph/badge.svg?token=NVFiiPydFA)](https://codecov.io/gh/ilabcode/ActionModels.jl)
[![License: GNU](https://img.shields.io/badge/License-GNU-yellow)](<https://www.gnu.org/licenses/>)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# Welcome to ActionModels!

ActionModels.jl is a Julia package for computational modeling of cognition and behaviour.
It can be used to fit cognitive models to data, as well as to simulate behaviour.
ActionModels allows for easy specification of hiearchical models, as well as including generalized linear regressions of model parameters, using standard LMER syntax.
ActionModels makes it easy to specify new models, but also contains a growing library of precreated models, and can easily be extended to include more complex model families.
The package is designed to have functions for every step of a classic cognitive modelling framework, such as parmeter recovery, predictive checks, and extracting trajectories of cognitive states for further analysis.
Inspired by packages like brms and HBayesDM, ActionModels is designed to provide a flexible but intuitive and easy-to-use interface for a field that is otherwise only accessible to technical experts.
Under the hood, ActionModels relies on Turing.jl, Julia's powerful framework for probabilistic modelling, but Julia's native automatic differentiation means that users do not have to engage directly with Turing's API.
ActionModels is continuously being developed and optimised within the constraints of cognitive modelling. It allows for parallelizing models across experimental sessions, and can use Turing's composite samplers to estimate both continuous and discrete parameters.
This documentation covers all three main components of ActionModels: defining cognitive models, fitting them to data, and simulating behaviour.
It also describes how to extend or contribute to ActionModels to include new models, and how to debug models. It beings, however, with a brief theoretical introduction to the field and method of cognitive modelling.

# Getting Started

First we load the ActionModels package

````julia
using ActionModels
````

We can now quickly define a cognitive model. We write a function that describes the action selection process in a single timestep.
Here we create the classic Rescorla-Wagner model, with a Gaussian-noise report as action:

````julia
function rescorla_wagner(attributes::ModelAttributes, observation::Float64)
    #Read in parameters and states
    parameters = load_parameters(attributes)
    states = load_states(attributes)

    α = parameters.learning_rate
    β = parameters.action_noise
    Vₜ₋₁ = states.expected_value

    #The Rescorla-Wagner update rule updates the expected value U, based on the observation and the learning rate α
    Vₜ = Vₜ₋₁ + α * (observation - Vₜ₋₁)

    #The updated expected value is stored to be accessed on next timestep
    update_state!(attributes, :expected_value, Vₜ)

    #The probability distribution for the action on this timestep is a Gaussian with the expected value V as mean, and a noise parameter β as standard deviation
    action_distribution = Normal(Vₜ, β)

    return action_distribution
end;
````

We now create the model object.
We first define the attributes of the Rescorla Wagner model. This includes it's three parameters, the expected value state, the observation and the action:
Then we use the ActionModel constructor to create the model object.

````julia
parameters = (
    learning_rate = Parameter(0.1),                             #The learning rate, with a default value of 0.1
    action_noise = Parameter(1),                                #The action noise, with a default value of 1
    initial_value = InitialStateParameter(0, :expected_value),  #And the initial expected value V₀, with a default value of 0
)
states = (;
    expected_value = State(),           #The expected value V, which is updated on each timestep
)
observations = (;
    observation = Observation()         #The observation, which is passed to the model on each timestep and used to update V
)
actions = (;
    report = Action(Normal)             #The report action, which reports the expected value with Gaussian noise
)

action_model = ActionModel(
    rescorla_wagner,
    parameters = parameters,
    states = states,
    observations = observations,
    actions = actions,
)
````

````
-- ActionModel --
Action model function: rescorla_wagner
Number of parameters: 3
Number of states: 1
Number of observations: 1
Number of actions: 1

````

We can now read in a dataset. In this example, we will use a simple simulated dataset, where three participants each have stated predictions after each of 6 observations, under some treatment condition as well as in a control condition.

````julia
using DataFrames

data = DataFrame(
    observations = repeat([1.0, 1, 1, 2, 2, 2], 6),
    actions = vcat(
        [0, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0, 0.5, 0.8, 1, 1.5, 1.8],
        [0, 2, 0.5, 4, 5, 3],
        [0, 0.1, 0.15, 0.2, 0.25, 0.3],
        [0, 0.2, 0.4, 0.7, 1.0, 1.1],
        [0, 2, 0.5, 4, 5, 3],
    ),
    id = vcat(
        repeat(["A"], 6),
        repeat(["B"], 6),
        repeat(["C"], 6),
        repeat(["A"], 6),
        repeat(["B"], 6),
        repeat(["C"], 6),
    ),
    treatment = vcat(repeat(["control"], 18), repeat(["treatment"], 18)),
)

show(data)
````

````
36×4 DataFrame
 Row │ observations  actions  id      treatment
     │ Float64       Float64  String  String
─────┼──────────────────────────────────────────
   1 │          1.0      0.0  A       control
   2 │          1.0      0.2  A       control
   3 │          1.0      0.3  A       control
   4 │          2.0      0.4  A       control
   5 │          2.0      0.5  A       control
   6 │          2.0      0.6  A       control
   7 │          1.0      0.0  B       control
   8 │          1.0      0.5  B       control
  ⋮  │      ⋮           ⋮       ⋮         ⋮
  30 │          2.0      1.1  B       treatment
  31 │          1.0      0.0  C       treatment
  32 │          1.0      2.0  C       treatment
  33 │          1.0      0.5  C       treatment
  34 │          2.0      4.0  C       treatment
  35 │          2.0      5.0  C       treatment
  36 │          2.0      3.0  C       treatment
                                 21 rows omitted
````

We can now create a model for estimating parameters hierarchically for each participant.
We make a regression model where we estimate how much the learning rate and action noise differ between treatment conditions.
We include a random intercept for each participant, making this a hierarchical model.
The initial value parameter is not estimated, and is fixed to it's default: 0.

````julia
model = create_model(
    action_model,
    [
        Regression(@formula(learning_rate ~ treatment + (1 | id)), logistic), #use a logistic link function to ensure that the learning rate is between 0 and 1
        Regression(@formula(action_noise ~ treatment + (1 | id)), exp),        #use an exponential link function to ensure that the action noise is positive
    ],
    data;
    action_cols = :actions,
    observation_cols = :observations,
    session_cols = [:id, :treatment],
)
````

````
-- ModelFit object --
Action model: rescorla_wagner
Linear regression population model
2 estimated action model parameters, 6 sessions
Posterior not sampled
Prior not sampled

````

We can now fit the model to the data, extract the estimated parameters for each participant, and summarize it as a dataframe:

````julia
using StatsPlots #load statsplots for plotting results

sample_posterior!(model, progress = false);                       #Fit the model to the data
parameters_per_session = get_session_parameters!(model)           #Extract the full distribution of parameters for each participant
summarized_parameters = summarize(parameters_per_session, median) #Populate a dataframe with the median of each posterior distribution

show(summarized_parameters)
````

````
┌ Info: Found initial step size
└   ϵ = 0.20078125000000002
┌ Info: Found initial step size
└   ϵ = 0.003125
6×4 DataFrame
 Row │ id      treatment  action_noise  learning_rate
     │ String  String     Float64       Float64
─────┼────────────────────────────────────────────────
   1 │ A       control       0.0537596      0.083392
   2 │ B       control       0.174333       0.32274
   3 │ C       control       2.19436        0.842449
   4 │ A       treatment     0.0374966      0.0381724
   5 │ B       treatment     0.120997       0.171628
   6 │ C       treatment     1.52999        0.697754
````

TODO: we can plot the estimated parameters

We can also extract the estimated value of V at each timestep, for each participant:

````julia
state_trajectories = get_state_trajectories!(model, :expected_value) #Extract the estimated trajectory of V
summarized_trajectories = summarize(state_trajectories, median)      #Summarize the trajectories

show(summarized_trajectories)
````

````
42×4 DataFrame
 Row │ id      treatment  timestep  expected_value
     │ String  String     Int64     Float64?
─────┼─────────────────────────────────────────────
   1 │ A       control           0        0.0
   2 │ A       control           1        0.083392
   3 │ A       control           2        0.15983
   4 │ A       control           3        0.229893
   5 │ A       control           4        0.377506
   6 │ A       control           5        0.512809
   7 │ A       control           6        0.636829
   8 │ B       control           0        0.0
  ⋮  │   ⋮         ⋮         ⋮            ⋮
  36 │ C       treatment         0        0.0
  37 │ C       treatment         1        0.697754
  38 │ C       treatment         2        0.908647
  39 │ C       treatment         3        0.972389
  40 │ C       treatment         4        1.68941
  41 │ C       treatment         5        1.90612
  42 │ C       treatment         6        1.97163
                                    27 rows omitted
````

TODO: we can also plot the estimated state trajectory

Finally, we can also simulate behaviour using the model.
First we instantiate an Agent object, which produces actions according to the action model.
Additionally, we can specify which states to save in the history of the agent.

````julia
agent = init_agent(action_model, save_history = [:expected_value]) #Create an agent object
````

````
-- ActionModels Agent --
Action model: rescorla_wagner
This agent has received 0 observations

````

We can set parameter values for the agent, and simulate behaviour for some set of observations

````julia
#Set the parameters of the agent
set_parameters!(agent, (learning_rate = 0.8, action_noise = 0.01))

#Simulate the agent for 6 timesteps, with some the specified observations
observations = [0.1, 0.1, 0.2, 0.3, 0.1, 0.2]
simulated_actions = simulate!(agent, observations)

#Plot the change in expected value over time
plot(agent, :expected_value, label = "expected value", ylabel = "value")
plot!(observations, linetype = :scatter, label = "observation")
plot!(simulated_actions, linetype = :scatter, label = "action")
````

```@raw html
<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="600" height="400" viewBox="0 0 2400 1600">
<defs>
  <clipPath id="clip100">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip100)" d="M0 1600 L2400 1600 L2400 8.88178e-14 L0 8.88178e-14  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip101">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip100)" d="M219.38 1423.18 L2352.76 1423.18 L2352.76 123.472 L219.38 123.472  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip102">
    <rect x="219" y="123" width="2134" height="1301"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip102)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="279.759,1423.18 279.759,123.472 "/>
<polyline clip-path="url(#clip102)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="615.195,1423.18 615.195,123.472 "/>
<polyline clip-path="url(#clip102)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="950.632,1423.18 950.632,123.472 "/>
<polyline clip-path="url(#clip102)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1286.07,1423.18 1286.07,123.472 "/>
<polyline clip-path="url(#clip102)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1621.5,1423.18 1621.5,123.472 "/>
<polyline clip-path="url(#clip102)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1956.94,1423.18 1956.94,123.472 "/>
<polyline clip-path="url(#clip102)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2292.38,1423.18 2292.38,123.472 "/>
<polyline clip-path="url(#clip102)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.38,1386.4 2352.76,1386.4 "/>
<polyline clip-path="url(#clip102)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.38,977.683 2352.76,977.683 "/>
<polyline clip-path="url(#clip102)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.38,568.969 2352.76,568.969 "/>
<polyline clip-path="url(#clip102)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.38,160.256 2352.76,160.256 "/>
<polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.38,1423.18 2352.76,1423.18 "/>
<polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="279.759,1423.18 279.759,1404.28 "/>
<polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="615.195,1423.18 615.195,1404.28 "/>
<polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="950.632,1423.18 950.632,1404.28 "/>
<polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1286.07,1423.18 1286.07,1404.28 "/>
<polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1621.5,1423.18 1621.5,1404.28 "/>
<polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1956.94,1423.18 1956.94,1404.28 "/>
<polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2292.38,1423.18 2292.38,1404.28 "/>
<path clip-path="url(#clip100)" d="M279.759 1454.1 Q276.148 1454.1 274.319 1457.66 Q272.513 1461.2 272.513 1468.33 Q272.513 1475.44 274.319 1479.01 Q276.148 1482.55 279.759 1482.55 Q283.393 1482.55 285.199 1479.01 Q287.027 1475.44 287.027 1468.33 Q287.027 1461.2 285.199 1457.66 Q283.393 1454.1 279.759 1454.1 M279.759 1450.39 Q285.569 1450.39 288.624 1455 Q291.703 1459.58 291.703 1468.33 Q291.703 1477.06 288.624 1481.67 Q285.569 1486.25 279.759 1486.25 Q273.949 1486.25 270.87 1481.67 Q267.814 1477.06 267.814 1468.33 Q267.814 1459.58 270.87 1455 Q273.949 1450.39 279.759 1450.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M605.577 1481.64 L613.216 1481.64 L613.216 1455.28 L604.906 1456.95 L604.906 1452.69 L613.17 1451.02 L617.846 1451.02 L617.846 1481.64 L625.484 1481.64 L625.484 1485.58 L605.577 1485.58 L605.577 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M945.284 1481.64 L961.604 1481.64 L961.604 1485.58 L939.659 1485.58 L939.659 1481.64 Q942.321 1478.89 946.905 1474.26 Q951.511 1469.61 952.692 1468.27 Q954.937 1465.74 955.817 1464.01 Q956.72 1462.25 956.72 1460.56 Q956.72 1457.8 954.775 1456.07 Q952.854 1454.33 949.752 1454.33 Q947.553 1454.33 945.099 1455.09 Q942.669 1455.86 939.891 1457.41 L939.891 1452.69 Q942.715 1451.55 945.169 1450.97 Q947.622 1450.39 949.659 1450.39 Q955.03 1450.39 958.224 1453.08 Q961.419 1455.77 961.419 1460.26 Q961.419 1462.39 960.608 1464.31 Q959.821 1466.2 957.715 1468.8 Q957.136 1469.47 954.034 1472.69 Q950.933 1475.88 945.284 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1290.32 1466.95 Q1293.67 1467.66 1295.55 1469.93 Q1297.45 1472.2 1297.45 1475.53 Q1297.45 1480.65 1293.93 1483.45 Q1290.41 1486.25 1283.93 1486.25 Q1281.75 1486.25 1279.44 1485.81 Q1277.14 1485.39 1274.69 1484.54 L1274.69 1480.02 Q1276.64 1481.16 1278.95 1481.74 Q1281.26 1482.32 1283.79 1482.32 Q1288.19 1482.32 1290.48 1480.58 Q1292.79 1478.84 1292.79 1475.53 Q1292.79 1472.48 1290.64 1470.77 Q1288.51 1469.03 1284.69 1469.03 L1280.66 1469.03 L1280.66 1465.19 L1284.88 1465.19 Q1288.32 1465.19 1290.15 1463.82 Q1291.98 1462.43 1291.98 1459.84 Q1291.98 1457.18 1290.08 1455.77 Q1288.21 1454.33 1284.69 1454.33 Q1282.77 1454.33 1280.57 1454.75 Q1278.37 1455.16 1275.73 1456.04 L1275.73 1451.88 Q1278.39 1451.14 1280.71 1450.77 Q1283.05 1450.39 1285.11 1450.39 Q1290.43 1450.39 1293.53 1452.83 Q1296.64 1455.23 1296.64 1459.35 Q1296.64 1462.22 1294.99 1464.21 Q1293.35 1466.18 1290.32 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1624.51 1455.09 L1612.71 1473.54 L1624.51 1473.54 L1624.51 1455.09 M1623.29 1451.02 L1629.17 1451.02 L1629.17 1473.54 L1634.1 1473.54 L1634.1 1477.43 L1629.17 1477.43 L1629.17 1485.58 L1624.51 1485.58 L1624.51 1477.43 L1608.91 1477.43 L1608.91 1472.92 L1623.29 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1947.22 1451.02 L1965.58 1451.02 L1965.58 1454.96 L1951.5 1454.96 L1951.5 1463.43 Q1952.52 1463.08 1953.54 1462.92 Q1954.56 1462.73 1955.58 1462.73 Q1961.36 1462.73 1964.74 1465.9 Q1968.12 1469.08 1968.12 1474.49 Q1968.12 1480.07 1964.65 1483.17 Q1961.18 1486.25 1954.86 1486.25 Q1952.68 1486.25 1950.41 1485.88 Q1948.17 1485.51 1945.76 1484.77 L1945.76 1480.07 Q1947.84 1481.2 1950.07 1481.76 Q1952.29 1482.32 1954.76 1482.32 Q1958.77 1482.32 1961.11 1480.21 Q1963.45 1478.1 1963.45 1474.49 Q1963.45 1470.88 1961.11 1468.77 Q1958.77 1466.67 1954.76 1466.67 Q1952.89 1466.67 1951.02 1467.08 Q1949.16 1467.5 1947.22 1468.38 L1947.22 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2292.78 1466.44 Q2289.63 1466.44 2287.78 1468.59 Q2285.95 1470.74 2285.95 1474.49 Q2285.95 1478.22 2287.78 1480.39 Q2289.63 1482.55 2292.78 1482.55 Q2295.93 1482.55 2297.76 1480.39 Q2299.61 1478.22 2299.61 1474.49 Q2299.61 1470.74 2297.76 1468.59 Q2295.93 1466.44 2292.78 1466.44 M2302.06 1451.78 L2302.06 1456.04 Q2300.31 1455.21 2298.5 1454.77 Q2296.72 1454.33 2294.96 1454.33 Q2290.33 1454.33 2287.88 1457.45 Q2285.44 1460.58 2285.1 1466.9 Q2286.46 1464.89 2288.52 1463.82 Q2290.58 1462.73 2293.06 1462.73 Q2298.27 1462.73 2301.28 1465.9 Q2304.31 1469.05 2304.31 1474.49 Q2304.31 1479.82 2301.16 1483.03 Q2298.01 1486.25 2292.78 1486.25 Q2286.79 1486.25 2283.62 1481.67 Q2280.44 1477.06 2280.44 1468.33 Q2280.44 1460.14 2284.33 1455.28 Q2288.22 1450.39 2294.77 1450.39 Q2296.53 1450.39 2298.31 1450.74 Q2300.12 1451.09 2302.06 1451.78 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1137.49 1520.52 L1177.69 1520.52 L1177.69 1525.93 L1160.82 1525.93 L1160.82 1568.04 L1154.36 1568.04 L1154.36 1525.93 L1137.49 1525.93 L1137.49 1520.52 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1181.64 1532.4 L1187.5 1532.4 L1187.5 1568.04 L1181.64 1568.04 L1181.64 1532.4 M1181.64 1518.52 L1187.5 1518.52 L1187.5 1525.93 L1181.64 1525.93 L1181.64 1518.52 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1227.5 1539.24 Q1229.7 1535.29 1232.76 1533.41 Q1235.81 1531.54 1239.95 1531.54 Q1245.52 1531.54 1248.54 1535.45 Q1251.57 1539.33 1251.57 1546.53 L1251.57 1568.04 L1245.68 1568.04 L1245.68 1546.72 Q1245.68 1541.59 1243.86 1539.11 Q1242.05 1536.63 1238.33 1536.63 Q1233.77 1536.63 1231.13 1539.65 Q1228.49 1542.68 1228.49 1547.9 L1228.49 1568.04 L1222.6 1568.04 L1222.6 1546.72 Q1222.6 1541.56 1220.79 1539.11 Q1218.97 1536.63 1215.19 1536.63 Q1210.7 1536.63 1208.06 1539.68 Q1205.41 1542.71 1205.41 1547.9 L1205.41 1568.04 L1199.53 1568.04 L1199.53 1532.4 L1205.41 1532.4 L1205.41 1537.93 Q1207.42 1534.66 1210.22 1533.1 Q1213.02 1531.54 1216.87 1531.54 Q1220.76 1531.54 1223.46 1533.51 Q1226.2 1535.48 1227.5 1539.24 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1293.74 1548.76 L1293.74 1551.62 L1266.81 1551.62 Q1267.19 1557.67 1270.44 1560.85 Q1273.72 1564 1279.54 1564 Q1282.92 1564 1286.07 1563.17 Q1289.25 1562.35 1292.37 1560.69 L1292.37 1566.23 Q1289.22 1567.57 1285.91 1568.27 Q1282.6 1568.97 1279.19 1568.97 Q1270.66 1568.97 1265.67 1564 Q1260.7 1559.04 1260.7 1550.57 Q1260.7 1541.82 1265.41 1536.69 Q1270.15 1531.54 1278.17 1531.54 Q1285.37 1531.54 1289.54 1536.18 Q1293.74 1540.8 1293.74 1548.76 M1287.88 1547.04 Q1287.82 1542.23 1285.18 1539.37 Q1282.57 1536.5 1278.24 1536.5 Q1273.34 1536.5 1270.38 1539.27 Q1267.45 1542.04 1267 1547.07 L1287.88 1547.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1326.08 1533.45 L1326.08 1538.98 Q1323.59 1537.71 1320.92 1537.07 Q1318.25 1536.44 1315.38 1536.44 Q1311.02 1536.44 1308.83 1537.77 Q1306.66 1539.11 1306.66 1541.79 Q1306.66 1543.82 1308.22 1545 Q1309.78 1546.15 1314.49 1547.2 L1316.5 1547.64 Q1322.73 1548.98 1325.34 1551.43 Q1327.99 1553.85 1327.99 1558.21 Q1327.99 1563.17 1324.04 1566.07 Q1320.12 1568.97 1313.25 1568.97 Q1310.39 1568.97 1307.27 1568.39 Q1304.18 1567.85 1300.74 1566.74 L1300.74 1560.69 Q1303.99 1562.38 1307.14 1563.24 Q1310.29 1564.07 1313.38 1564.07 Q1317.51 1564.07 1319.74 1562.66 Q1321.97 1561.23 1321.97 1558.65 Q1321.97 1556.27 1320.35 1554.99 Q1318.76 1553.72 1313.31 1552.54 L1311.28 1552.07 Q1305.83 1550.92 1303.41 1548.56 Q1301 1546.18 1301 1542.04 Q1301 1537.01 1304.56 1534.27 Q1308.13 1531.54 1314.68 1531.54 Q1317.93 1531.54 1320.79 1532.01 Q1323.66 1532.49 1326.08 1533.45 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1343.1 1522.27 L1343.1 1532.4 L1355.17 1532.4 L1355.17 1536.95 L1343.1 1536.95 L1343.1 1556.3 Q1343.1 1560.66 1344.28 1561.9 Q1345.49 1563.14 1349.15 1563.14 L1355.17 1563.14 L1355.17 1568.04 L1349.15 1568.04 Q1342.37 1568.04 1339.79 1565.53 Q1337.22 1562.98 1337.22 1556.3 L1337.22 1536.95 L1332.92 1536.95 L1332.92 1532.4 L1337.22 1532.4 L1337.22 1522.27 L1343.1 1522.27 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1393.36 1548.76 L1393.36 1551.62 L1366.44 1551.62 Q1366.82 1557.67 1370.06 1560.85 Q1373.34 1564 1379.17 1564 Q1382.54 1564 1385.69 1563.17 Q1388.87 1562.35 1391.99 1560.69 L1391.99 1566.23 Q1388.84 1567.57 1385.53 1568.27 Q1382.22 1568.97 1378.82 1568.97 Q1370.29 1568.97 1365.29 1564 Q1360.32 1559.04 1360.32 1550.57 Q1360.32 1541.82 1365.03 1536.69 Q1369.78 1531.54 1377.8 1531.54 Q1384.99 1531.54 1389.16 1536.18 Q1393.36 1540.8 1393.36 1548.76 M1387.51 1547.04 Q1387.44 1542.23 1384.8 1539.37 Q1382.19 1536.5 1377.86 1536.5 Q1372.96 1536.5 1370 1539.27 Q1367.07 1542.04 1366.63 1547.07 L1387.51 1547.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1408.64 1562.7 L1408.64 1581.6 L1402.75 1581.6 L1402.75 1532.4 L1408.64 1532.4 L1408.64 1537.81 Q1410.49 1534.62 1413.29 1533.1 Q1416.12 1531.54 1420.03 1531.54 Q1426.53 1531.54 1430.57 1536.69 Q1434.64 1541.85 1434.64 1550.25 Q1434.64 1558.65 1430.57 1563.81 Q1426.53 1568.97 1420.03 1568.97 Q1416.12 1568.97 1413.29 1567.44 Q1410.49 1565.88 1408.64 1562.7 M1428.56 1550.25 Q1428.56 1543.79 1425.89 1540.13 Q1423.25 1536.44 1418.6 1536.44 Q1413.96 1536.44 1411.28 1540.13 Q1408.64 1543.79 1408.64 1550.25 Q1408.64 1556.71 1411.28 1560.4 Q1413.96 1564.07 1418.6 1564.07 Q1423.25 1564.07 1425.89 1560.4 Q1428.56 1556.71 1428.56 1550.25 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.38,1423.18 219.38,123.472 "/>
<polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.38,1386.4 238.278,1386.4 "/>
<polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.38,977.683 238.278,977.683 "/>
<polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.38,568.969 238.278,568.969 "/>
<polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.38,160.256 238.278,160.256 "/>
<path clip-path="url(#clip100)" d="M126.205 1372.19 Q122.593 1372.19 120.765 1375.76 Q118.959 1379.3 118.959 1386.43 Q118.959 1393.54 120.765 1397.1 Q122.593 1400.64 126.205 1400.64 Q129.839 1400.64 131.644 1397.1 Q133.473 1393.54 133.473 1386.43 Q133.473 1379.3 131.644 1375.76 Q129.839 1372.19 126.205 1372.19 M126.205 1368.49 Q132.015 1368.49 135.07 1373.1 Q138.149 1377.68 138.149 1386.43 Q138.149 1395.16 135.07 1399.76 Q132.015 1404.35 126.205 1404.35 Q120.394 1404.35 117.316 1399.76 Q114.26 1395.16 114.26 1386.43 Q114.26 1377.68 117.316 1373.1 Q120.394 1368.49 126.205 1368.49 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M146.366 1397.8 L151.251 1397.8 L151.251 1403.68 L146.366 1403.68 L146.366 1397.8 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M171.436 1372.19 Q167.825 1372.19 165.996 1375.76 Q164.19 1379.3 164.19 1386.43 Q164.19 1393.54 165.996 1397.1 Q167.825 1400.64 171.436 1400.64 Q175.07 1400.64 176.876 1397.1 Q178.704 1393.54 178.704 1386.43 Q178.704 1379.3 176.876 1375.76 Q175.07 1372.19 171.436 1372.19 M171.436 1368.49 Q177.246 1368.49 180.301 1373.1 Q183.38 1377.68 183.38 1386.43 Q183.38 1395.16 180.301 1399.76 Q177.246 1404.35 171.436 1404.35 Q165.626 1404.35 162.547 1399.76 Q159.491 1395.16 159.491 1386.43 Q159.491 1377.68 162.547 1373.1 Q165.626 1368.49 171.436 1368.49 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M127.431 963.481 Q123.82 963.481 121.992 967.046 Q120.186 970.588 120.186 977.717 Q120.186 984.824 121.992 988.389 Q123.82 991.93 127.431 991.93 Q131.066 991.93 132.871 988.389 Q134.7 984.824 134.7 977.717 Q134.7 970.588 132.871 967.046 Q131.066 963.481 127.431 963.481 M127.431 959.778 Q133.242 959.778 136.297 964.384 Q139.376 968.967 139.376 977.717 Q139.376 986.444 136.297 991.051 Q133.242 995.634 127.431 995.634 Q121.621 995.634 118.543 991.051 Q115.487 986.444 115.487 977.717 Q115.487 968.967 118.543 964.384 Q121.621 959.778 127.431 959.778 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M147.593 989.083 L152.478 989.083 L152.478 994.963 L147.593 994.963 L147.593 989.083 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M163.473 991.027 L171.112 991.027 L171.112 964.662 L162.802 966.328 L162.802 962.069 L171.065 960.403 L175.741 960.403 L175.741 991.027 L183.38 991.027 L183.38 994.963 L163.473 994.963 L163.473 991.027 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M127.802 554.768 Q124.191 554.768 122.362 558.333 Q120.556 561.875 120.556 569.004 Q120.556 576.111 122.362 579.675 Q124.191 583.217 127.802 583.217 Q131.436 583.217 133.242 579.675 Q135.07 576.111 135.07 569.004 Q135.07 561.875 133.242 558.333 Q131.436 554.768 127.802 554.768 M127.802 551.064 Q133.612 551.064 136.667 555.671 Q139.746 560.254 139.746 569.004 Q139.746 577.731 136.667 582.337 Q133.612 586.921 127.802 586.921 Q121.992 586.921 118.913 582.337 Q115.857 577.731 115.857 569.004 Q115.857 560.254 118.913 555.671 Q121.992 551.064 127.802 551.064 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M147.964 580.37 L152.848 580.37 L152.848 586.249 L147.964 586.249 L147.964 580.37 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M167.061 582.314 L183.38 582.314 L183.38 586.249 L161.436 586.249 L161.436 582.314 Q164.098 579.56 168.681 574.93 Q173.288 570.277 174.468 568.935 Q176.714 566.412 177.593 564.675 Q178.496 562.916 178.496 561.226 Q178.496 558.472 176.551 556.736 Q174.63 555 171.528 555 Q169.329 555 166.876 555.763 Q164.445 556.527 161.667 558.078 L161.667 553.356 Q164.491 552.222 166.945 551.643 Q169.399 551.064 171.436 551.064 Q176.806 551.064 180.001 553.75 Q183.195 556.435 183.195 560.925 Q183.195 563.055 182.385 564.976 Q181.598 566.875 179.491 569.467 Q178.913 570.138 175.811 573.356 Q172.709 576.55 167.061 582.314 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M126.853 146.055 Q123.242 146.055 121.413 149.62 Q119.607 153.161 119.607 160.291 Q119.607 167.397 121.413 170.962 Q123.242 174.504 126.853 174.504 Q130.487 174.504 132.292 170.962 Q134.121 167.397 134.121 160.291 Q134.121 153.161 132.292 149.62 Q130.487 146.055 126.853 146.055 M126.853 142.351 Q132.663 142.351 135.718 146.958 Q138.797 151.541 138.797 160.291 Q138.797 169.018 135.718 173.624 Q132.663 178.208 126.853 178.208 Q121.043 178.208 117.964 173.624 Q114.908 169.018 114.908 160.291 Q114.908 151.541 117.964 146.958 Q121.043 142.351 126.853 142.351 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M147.015 171.657 L151.899 171.657 L151.899 177.536 L147.015 177.536 L147.015 171.657 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M176.251 158.902 Q179.607 159.62 181.482 161.888 Q183.38 164.157 183.38 167.49 Q183.38 172.606 179.862 175.407 Q176.343 178.208 169.862 178.208 Q167.686 178.208 165.371 177.768 Q163.079 177.351 160.626 176.495 L160.626 171.981 Q162.57 173.115 164.885 173.694 Q167.2 174.272 169.723 174.272 Q174.121 174.272 176.413 172.536 Q178.727 170.8 178.727 167.49 Q178.727 164.435 176.575 162.722 Q174.445 160.985 170.626 160.985 L166.598 160.985 L166.598 157.143 L170.811 157.143 Q174.26 157.143 176.089 155.777 Q177.917 154.388 177.917 151.796 Q177.917 149.134 176.019 147.722 Q174.144 146.286 170.626 146.286 Q168.704 146.286 166.505 146.703 Q164.306 147.12 161.667 147.999 L161.667 143.833 Q164.329 143.092 166.644 142.722 Q168.982 142.351 171.042 142.351 Q176.366 142.351 179.468 144.782 Q182.57 147.189 182.57 151.31 Q182.57 154.18 180.926 156.171 Q179.283 158.138 176.251 158.902 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M28.3562 859.645 L28.3562 853.438 L58.275 842.298 L28.3562 831.158 L28.3562 824.952 L64.0042 838.32 L64.0042 846.277 L28.3562 859.645 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M46.0847 800.667 Q46.0847 807.764 47.7079 810.502 Q49.3312 813.239 53.2461 813.239 Q56.3653 813.239 58.2114 811.202 Q60.0256 809.133 60.0256 805.6 Q60.0256 800.73 56.5881 797.802 Q53.1188 794.842 47.3897 794.842 L46.0847 794.842 L46.0847 800.667 M43.6657 788.986 L64.0042 788.986 L64.0042 794.842 L58.5933 794.842 Q61.8398 796.847 63.3994 799.839 Q64.9272 802.831 64.9272 807.16 Q64.9272 812.634 61.8716 815.881 Q58.7843 819.095 53.6281 819.095 Q47.6125 819.095 44.5569 815.085 Q41.5014 811.043 41.5014 803.054 L41.5014 794.842 L40.9285 794.842 Q36.8862 794.842 34.6901 797.516 Q32.4621 800.157 32.4621 804.964 Q32.4621 808.019 33.1941 810.915 Q33.9262 813.812 35.3903 816.485 L29.9795 816.485 Q28.7381 813.271 28.1334 810.247 Q27.4968 807.223 27.4968 804.359 Q27.4968 796.624 31.5072 792.805 Q35.5176 788.986 43.6657 788.986 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M14.479 776.923 L14.479 771.066 L64.0042 771.066 L64.0042 776.923 L14.479 776.923 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M49.9359 759.417 L28.3562 759.417 L28.3562 753.56 L49.7131 753.56 Q54.7739 753.56 57.3202 751.587 Q59.8346 749.614 59.8346 745.667 Q59.8346 740.925 56.8109 738.187 Q53.7872 735.418 48.5673 735.418 L28.3562 735.418 L28.3562 729.562 L64.0042 729.562 L64.0042 735.418 L58.5296 735.418 Q61.7762 737.551 63.3676 740.383 Q64.9272 743.184 64.9272 746.908 Q64.9272 753.051 61.1078 756.234 Q57.2883 759.417 49.9359 759.417 M27.4968 744.68 L27.4968 744.68 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M44.7161 687.007 L47.5806 687.007 L47.5806 713.934 Q53.6281 713.552 56.8109 710.305 Q59.9619 707.027 59.9619 701.203 Q59.9619 697.829 59.1344 694.678 Q58.3069 691.495 56.6518 688.376 L62.1899 688.376 Q63.5267 691.527 64.227 694.837 Q64.9272 698.147 64.9272 701.553 Q64.9272 710.083 59.9619 715.08 Q54.9967 720.045 46.5303 720.045 Q37.7774 720.045 32.6531 715.334 Q27.4968 710.592 27.4968 702.571 Q27.4968 695.378 32.1438 691.208 Q36.7589 687.007 44.7161 687.007 M42.9973 692.863 Q38.1912 692.927 35.3266 695.569 Q32.4621 698.179 32.4621 702.507 Q32.4621 707.409 35.2312 710.369 Q38.0002 713.297 43.0292 713.743 L42.9973 692.863 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1006.45 48.0275 L1006.45 51.6733 L972.184 51.6733 Q972.67 59.3701 976.802 63.421 Q980.974 67.4314 988.387 67.4314 Q992.681 67.4314 996.692 66.3781 Q1000.74 65.3249 1004.71 63.2184 L1004.71 70.267 Q1000.7 71.9684 996.489 72.8596 Q992.276 73.7508 987.942 73.7508 Q977.085 73.7508 970.725 67.4314 Q964.406 61.1119 964.406 50.3365 Q964.406 39.1965 970.401 32.6746 Q976.437 26.1121 986.645 26.1121 Q995.8 26.1121 1001.11 32.0264 Q1006.45 37.9003 1006.45 48.0275 M999.001 45.84 Q998.92 39.7232 995.557 36.0774 Q992.236 32.4315 986.726 32.4315 Q980.488 32.4315 976.721 35.9558 Q972.994 39.4801 972.427 45.8805 L999.001 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1054.94 27.2059 L1038.54 49.2833 L1055.79 72.576 L1047 72.576 L1033.8 54.752 L1020.59 72.576 L1011.8 72.576 L1029.42 48.8377 L1013.3 27.2059 L1022.09 27.2059 L1034.12 43.369 L1046.15 27.2059 L1054.94 27.2059 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1073.54 65.7705 L1073.54 89.8329 L1066.04 89.8329 L1066.04 27.2059 L1073.54 27.2059 L1073.54 34.0924 Q1075.89 30.0415 1079.45 28.0971 Q1083.06 26.1121 1088.04 26.1121 Q1096.3 26.1121 1101.45 32.6746 Q1106.63 39.2371 1106.63 49.9314 Q1106.63 60.6258 1101.45 67.1883 Q1096.3 73.7508 1088.04 73.7508 Q1083.06 73.7508 1079.45 71.8063 Q1075.89 69.8214 1073.54 65.7705 M1098.9 49.9314 Q1098.9 41.7081 1095.49 37.0496 Q1092.13 32.3505 1086.22 32.3505 Q1080.3 32.3505 1076.9 37.0496 Q1073.54 41.7081 1073.54 49.9314 Q1073.54 58.1548 1076.9 62.8538 Q1080.3 67.5124 1086.22 67.5124 Q1092.13 67.5124 1095.49 62.8538 Q1098.9 58.1548 1098.9 49.9314 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1157.8 48.0275 L1157.8 51.6733 L1123.53 51.6733 Q1124.01 59.3701 1128.14 63.421 Q1132.32 67.4314 1139.73 67.4314 Q1144.02 67.4314 1148.03 66.3781 Q1152.08 65.3249 1156.05 63.2184 L1156.05 70.267 Q1152.04 71.9684 1147.83 72.8596 Q1143.62 73.7508 1139.28 73.7508 Q1128.43 73.7508 1122.07 67.4314 Q1115.75 61.1119 1115.75 50.3365 Q1115.75 39.1965 1121.74 32.6746 Q1127.78 26.1121 1137.99 26.1121 Q1147.14 26.1121 1152.45 32.0264 Q1157.8 37.9003 1157.8 48.0275 M1150.34 45.84 Q1150.26 39.7232 1146.9 36.0774 Q1143.58 32.4315 1138.07 32.4315 Q1131.83 32.4315 1128.06 35.9558 Q1124.34 39.4801 1123.77 45.8805 L1150.34 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1202.68 28.9478 L1202.68 35.9153 Q1199.52 34.1734 1196.32 33.3227 Q1193.16 32.4315 1189.92 32.4315 Q1182.67 32.4315 1178.66 37.0496 Q1174.65 41.6271 1174.65 49.9314 Q1174.65 58.2358 1178.66 62.8538 Q1182.67 67.4314 1189.92 67.4314 Q1193.16 67.4314 1196.32 66.5807 Q1199.52 65.6895 1202.68 63.9476 L1202.68 70.8341 Q1199.56 72.2924 1196.2 73.0216 Q1192.88 73.7508 1189.11 73.7508 Q1178.86 73.7508 1172.83 67.3098 Q1166.79 60.8689 1166.79 49.9314 Q1166.79 38.832 1172.87 32.472 Q1178.98 26.1121 1189.6 26.1121 Q1193.04 26.1121 1196.32 26.8413 Q1199.6 27.5299 1202.68 28.9478 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1223.02 14.324 L1223.02 27.2059 L1238.37 27.2059 L1238.37 32.9987 L1223.02 32.9987 L1223.02 57.6282 Q1223.02 63.1779 1224.51 64.7578 Q1226.05 66.3376 1230.71 66.3376 L1238.37 66.3376 L1238.37 72.576 L1230.71 72.576 Q1222.08 72.576 1218.8 69.3758 Q1215.52 66.1351 1215.52 57.6282 L1215.52 32.9987 L1210.05 32.9987 L1210.05 27.2059 L1215.52 27.2059 L1215.52 14.324 L1223.02 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1286.98 48.0275 L1286.98 51.6733 L1252.71 51.6733 Q1253.19 59.3701 1257.33 63.421 Q1261.5 67.4314 1268.91 67.4314 Q1273.21 67.4314 1277.22 66.3781 Q1281.27 65.3249 1285.24 63.2184 L1285.24 70.267 Q1281.23 71.9684 1277.01 72.8596 Q1272.8 73.7508 1268.47 73.7508 Q1257.61 73.7508 1251.25 67.4314 Q1244.93 61.1119 1244.93 50.3365 Q1244.93 39.1965 1250.93 32.6746 Q1256.96 26.1121 1267.17 26.1121 Q1276.33 26.1121 1281.63 32.0264 Q1286.98 37.9003 1286.98 48.0275 M1279.53 45.84 Q1279.44 39.7232 1276.08 36.0774 Q1272.76 32.4315 1267.25 32.4315 Q1261.01 32.4315 1257.25 35.9558 Q1253.52 39.4801 1252.95 45.8805 L1279.53 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1329.07 34.0924 L1329.07 9.54393 L1336.52 9.54393 L1336.52 72.576 L1329.07 72.576 L1329.07 65.7705 Q1326.72 69.8214 1323.11 71.8063 Q1319.55 73.7508 1314.53 73.7508 Q1306.3 73.7508 1301.12 67.1883 Q1295.97 60.6258 1295.97 49.9314 Q1295.97 39.2371 1301.12 32.6746 Q1306.3 26.1121 1314.53 26.1121 Q1319.55 26.1121 1323.11 28.0971 Q1326.72 30.0415 1329.07 34.0924 M1303.67 49.9314 Q1303.67 58.1548 1307.03 62.8538 Q1310.43 67.5124 1316.35 67.5124 Q1322.26 67.5124 1325.67 62.8538 Q1329.07 58.1548 1329.07 49.9314 Q1329.07 41.7081 1325.67 37.0496 Q1322.26 32.3505 1316.35 32.3505 Q1310.43 32.3505 1307.03 37.0496 Q1303.67 41.7081 1303.67 49.9314 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1386.35 86.3491 L1386.35 92.1419 L1343.25 92.1419 L1343.25 86.3491 L1386.35 86.3491 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1388.01 27.2059 L1395.91 27.2059 L1410.09 65.2844 L1424.26 27.2059 L1432.16 27.2059 L1415.15 72.576 L1405.02 72.576 L1388.01 27.2059 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1463.07 49.7694 Q1454.04 49.7694 1450.55 51.8354 Q1447.07 53.9013 1447.07 58.8839 Q1447.07 62.8538 1449.66 65.2034 Q1452.3 67.5124 1456.79 67.5124 Q1462.99 67.5124 1466.72 63.1374 Q1470.49 58.7219 1470.49 51.4303 L1470.49 49.7694 L1463.07 49.7694 M1477.94 46.6907 L1477.94 72.576 L1470.49 72.576 L1470.49 65.6895 Q1467.93 69.8214 1464.13 71.8063 Q1460.32 73.7508 1454.81 73.7508 Q1447.84 73.7508 1443.71 69.8619 Q1439.62 65.9325 1439.62 59.3701 Q1439.62 51.7138 1444.72 47.825 Q1449.87 43.9361 1460.03 43.9361 L1470.49 43.9361 L1470.49 43.2069 Q1470.49 38.0623 1467.08 35.2672 Q1463.72 32.4315 1457.6 32.4315 Q1453.71 32.4315 1450.03 33.3632 Q1446.34 34.295 1442.94 36.1584 L1442.94 29.2718 Q1447.03 27.692 1450.88 26.9223 Q1454.73 26.1121 1458.37 26.1121 Q1468.22 26.1121 1473.08 31.2163 Q1477.94 36.3204 1477.94 46.6907 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1493.29 9.54393 L1500.75 9.54393 L1500.75 72.576 L1493.29 72.576 L1493.29 9.54393 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1515.57 54.671 L1515.57 27.2059 L1523.03 27.2059 L1523.03 54.3874 Q1523.03 60.8284 1525.54 64.0691 Q1528.05 67.2693 1533.07 67.2693 Q1539.11 67.2693 1542.59 63.421 Q1546.12 59.5726 1546.12 52.9291 L1546.12 27.2059 L1553.57 27.2059 L1553.57 72.576 L1546.12 72.576 L1546.12 65.6084 Q1543.4 69.7404 1539.8 71.7658 Q1536.23 73.7508 1531.49 73.7508 Q1523.67 73.7508 1519.62 68.8897 Q1515.57 64.0286 1515.57 54.671 M1534.33 26.1121 L1534.33 26.1121 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1607.73 48.0275 L1607.73 51.6733 L1573.46 51.6733 Q1573.95 59.3701 1578.08 63.421 Q1582.25 67.4314 1589.66 67.4314 Q1593.96 67.4314 1597.97 66.3781 Q1602.02 65.3249 1605.99 63.2184 L1605.99 70.267 Q1601.98 71.9684 1597.76 72.8596 Q1593.55 73.7508 1589.22 73.7508 Q1578.36 73.7508 1572 67.4314 Q1565.68 61.1119 1565.68 50.3365 Q1565.68 39.1965 1571.68 32.6746 Q1577.71 26.1121 1587.92 26.1121 Q1597.08 26.1121 1602.38 32.0264 Q1607.73 37.9003 1607.73 48.0275 M1600.28 45.84 Q1600.2 39.7232 1596.83 36.0774 Q1593.51 32.4315 1588 32.4315 Q1581.76 32.4315 1578 35.9558 Q1574.27 39.4801 1573.7 45.8805 L1600.28 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip102)" style="stroke:#009af9; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="279.759,1386.4 615.195,1059.43 950.632,994.031 1286.07,653.982 1621.5,259.001 1956.94,833.946 2292.38,621.965 "/>
<circle clip-path="url(#clip102)" cx="615.195" cy="977.683" r="14.4" fill="#e26f46" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip102)" cx="950.632" cy="977.683" r="14.4" fill="#e26f46" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip102)" cx="1286.07" cy="568.969" r="14.4" fill="#e26f46" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip102)" cx="1621.5" cy="160.256" r="14.4" fill="#e26f46" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip102)" cx="1956.94" cy="977.683" r="14.4" fill="#e26f46" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip102)" cx="2292.38" cy="568.969" r="14.4" fill="#e26f46" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip102)" cx="615.195" cy="1081.86" r="14.4" fill="#3da44d" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip102)" cx="950.632" cy="1006.41" r="14.4" fill="#3da44d" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip102)" cx="1286.07" cy="559.214" r="14.4" fill="#3da44d" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip102)" cx="1621.5" cy="201.819" r="14.4" fill="#3da44d" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip102)" cx="1956.94" cy="847.646" r="14.4" fill="#3da44d" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip102)" cx="2292.38" cy="661.069" r="14.4" fill="#3da44d" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<path clip-path="url(#clip100)" d="M1709.33 374.156 L2281.64 374.156 L2281.64 166.796 L1709.33 166.796  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<polyline clip-path="url(#clip100)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1709.33,374.156 2281.64,374.156 2281.64,166.796 1709.33,166.796 1709.33,374.156 "/>
<polyline clip-path="url(#clip100)" style="stroke:#009af9; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1733.03,218.636 1875.26,218.636 "/>
<path clip-path="url(#clip100)" d="M1922.99 221.888 L1922.99 223.971 L1903.4 223.971 Q1903.68 228.369 1906.04 230.684 Q1908.43 232.976 1912.66 232.976 Q1915.12 232.976 1917.41 232.374 Q1919.72 231.772 1921.99 230.569 L1921.99 234.596 Q1919.7 235.568 1917.29 236.078 Q1914.89 236.587 1912.41 236.587 Q1906.2 236.587 1902.57 232.976 Q1898.96 229.365 1898.96 223.207 Q1898.96 216.842 1902.39 213.115 Q1905.83 209.365 1911.67 209.365 Q1916.9 209.365 1919.93 212.745 Q1922.99 216.101 1922.99 221.888 M1918.73 220.638 Q1918.68 217.143 1916.76 215.059 Q1914.86 212.976 1911.71 212.976 Q1908.15 212.976 1906 214.99 Q1903.87 217.004 1903.54 220.661 L1918.73 220.638 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1950.7 209.99 L1941.32 222.606 L1951.18 235.916 L1946.16 235.916 L1938.61 225.731 L1931.07 235.916 L1926.04 235.916 L1936.11 222.351 L1926.9 209.99 L1931.92 209.99 L1938.8 219.226 L1945.67 209.99 L1950.7 209.99 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1961.32 232.027 L1961.32 245.777 L1957.04 245.777 L1957.04 209.99 L1961.32 209.99 L1961.32 213.925 Q1962.66 211.61 1964.7 210.499 Q1966.76 209.365 1969.61 209.365 Q1974.33 209.365 1977.27 213.115 Q1980.23 216.865 1980.23 222.976 Q1980.23 229.087 1977.27 232.837 Q1974.33 236.587 1969.61 236.587 Q1966.76 236.587 1964.7 235.476 Q1962.66 234.342 1961.32 232.027 M1975.81 222.976 Q1975.81 218.277 1973.87 215.615 Q1971.95 212.93 1968.57 212.93 Q1965.19 212.93 1963.24 215.615 Q1961.32 218.277 1961.32 222.976 Q1961.32 227.675 1963.24 230.36 Q1965.19 233.022 1968.57 233.022 Q1971.95 233.022 1973.87 230.36 Q1975.81 227.675 1975.81 222.976 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2009.47 221.888 L2009.47 223.971 L1989.89 223.971 Q1990.16 228.369 1992.52 230.684 Q1994.91 232.976 1999.14 232.976 Q2001.6 232.976 2003.89 232.374 Q2006.2 231.772 2008.47 230.569 L2008.47 234.596 Q2006.18 235.568 2003.77 236.078 Q2001.37 236.587 1998.89 236.587 Q1992.69 236.587 1989.05 232.976 Q1985.44 229.365 1985.44 223.207 Q1985.44 216.842 1988.87 213.115 Q1992.32 209.365 1998.15 209.365 Q2003.38 209.365 2006.41 212.745 Q2009.47 216.101 2009.47 221.888 M2005.21 220.638 Q2005.16 217.143 2003.24 215.059 Q2001.34 212.976 1998.2 212.976 Q1994.63 212.976 1992.48 214.99 Q1990.35 217.004 1990.02 220.661 L2005.21 220.638 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2035.12 210.985 L2035.12 214.967 Q2033.31 213.971 2031.48 213.485 Q2029.68 212.976 2027.82 212.976 Q2023.68 212.976 2021.39 215.615 Q2019.1 218.231 2019.1 222.976 Q2019.1 227.721 2021.39 230.36 Q2023.68 232.976 2027.82 232.976 Q2029.68 232.976 2031.48 232.49 Q2033.31 231.981 2035.12 230.985 L2035.12 234.92 Q2033.33 235.754 2031.41 236.17 Q2029.51 236.587 2027.36 236.587 Q2021.51 236.587 2018.06 232.906 Q2014.61 229.226 2014.61 222.976 Q2014.61 216.633 2018.08 212.999 Q2021.57 209.365 2027.64 209.365 Q2029.61 209.365 2031.48 209.782 Q2033.36 210.175 2035.12 210.985 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2046.74 202.629 L2046.74 209.99 L2055.51 209.99 L2055.51 213.3 L2046.74 213.3 L2046.74 227.374 Q2046.74 230.545 2047.59 231.448 Q2048.47 232.351 2051.13 232.351 L2055.51 232.351 L2055.51 235.916 L2051.13 235.916 Q2046.2 235.916 2044.33 234.087 Q2042.45 232.235 2042.45 227.374 L2042.45 213.3 L2039.33 213.3 L2039.33 209.99 L2042.45 209.99 L2042.45 202.629 L2046.74 202.629 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2083.29 221.888 L2083.29 223.971 L2063.7 223.971 Q2063.98 228.369 2066.34 230.684 Q2068.73 232.976 2072.96 232.976 Q2075.42 232.976 2077.71 232.374 Q2080.02 231.772 2082.29 230.569 L2082.29 234.596 Q2080 235.568 2077.59 236.078 Q2075.19 236.587 2072.71 236.587 Q2066.51 236.587 2062.87 232.976 Q2059.26 229.365 2059.26 223.207 Q2059.26 216.842 2062.69 213.115 Q2066.13 209.365 2071.97 209.365 Q2077.2 209.365 2080.23 212.745 Q2083.29 216.101 2083.29 221.888 M2079.03 220.638 Q2078.98 217.143 2077.06 215.059 Q2075.16 212.976 2072.01 212.976 Q2068.45 212.976 2066.3 214.99 Q2064.17 217.004 2063.84 220.661 L2079.03 220.638 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2107.34 213.925 L2107.34 199.897 L2111.6 199.897 L2111.6 235.916 L2107.34 235.916 L2107.34 232.027 Q2106 234.342 2103.94 235.476 Q2101.9 236.587 2099.03 236.587 Q2094.33 236.587 2091.37 232.837 Q2088.43 229.087 2088.43 222.976 Q2088.43 216.865 2091.37 213.115 Q2094.33 209.365 2099.03 209.365 Q2101.9 209.365 2103.94 210.499 Q2106 211.61 2107.34 213.925 M2092.82 222.976 Q2092.82 227.675 2094.75 230.36 Q2096.69 233.022 2100.07 233.022 Q2103.45 233.022 2105.39 230.36 Q2107.34 227.675 2107.34 222.976 Q2107.34 218.277 2105.39 215.615 Q2103.45 212.93 2100.07 212.93 Q2096.69 212.93 2094.75 215.615 Q2092.82 218.277 2092.82 222.976 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2132.38 209.99 L2136.9 209.99 L2145 231.749 L2153.1 209.99 L2157.62 209.99 L2147.89 235.916 L2142.11 235.916 L2132.38 209.99 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2175.28 222.883 Q2170.12 222.883 2168.12 224.064 Q2166.13 225.244 2166.13 228.092 Q2166.13 230.36 2167.62 231.703 Q2169.12 233.022 2171.69 233.022 Q2175.23 233.022 2177.36 230.522 Q2179.51 227.999 2179.51 223.832 L2179.51 222.883 L2175.28 222.883 M2183.77 221.124 L2183.77 235.916 L2179.51 235.916 L2179.51 231.981 Q2178.06 234.342 2175.88 235.476 Q2173.7 236.587 2170.56 236.587 Q2166.57 236.587 2164.21 234.365 Q2161.87 232.119 2161.87 228.369 Q2161.87 223.994 2164.79 221.772 Q2167.73 219.55 2173.54 219.55 L2179.51 219.55 L2179.51 219.133 Q2179.51 216.194 2177.57 214.596 Q2175.65 212.976 2172.15 212.976 Q2169.93 212.976 2167.82 213.508 Q2165.72 214.041 2163.77 215.106 L2163.77 211.17 Q2166.11 210.268 2168.31 209.828 Q2170.51 209.365 2172.59 209.365 Q2178.22 209.365 2181 212.282 Q2183.77 215.198 2183.77 221.124 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2192.55 199.897 L2196.81 199.897 L2196.81 235.916 L2192.55 235.916 L2192.55 199.897 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2205.28 225.684 L2205.28 209.99 L2209.54 209.99 L2209.54 225.522 Q2209.54 229.203 2210.97 231.055 Q2212.41 232.883 2215.28 232.883 Q2218.73 232.883 2220.72 230.684 Q2222.73 228.485 2222.73 224.689 L2222.73 209.99 L2226.99 209.99 L2226.99 235.916 L2222.73 235.916 L2222.73 231.934 Q2221.18 234.295 2219.12 235.453 Q2217.08 236.587 2214.37 236.587 Q2209.91 236.587 2207.59 233.809 Q2205.28 231.031 2205.28 225.684 M2215.99 209.365 L2215.99 209.365 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2257.94 221.888 L2257.94 223.971 L2238.36 223.971 Q2238.63 228.369 2240.99 230.684 Q2243.38 232.976 2247.62 232.976 Q2250.07 232.976 2252.36 232.374 Q2254.68 231.772 2256.94 230.569 L2256.94 234.596 Q2254.65 235.568 2252.24 236.078 Q2249.84 236.587 2247.36 236.587 Q2241.16 236.587 2237.52 232.976 Q2233.91 229.365 2233.91 223.207 Q2233.91 216.842 2237.34 213.115 Q2240.79 209.365 2246.62 209.365 Q2251.85 209.365 2254.88 212.745 Q2257.94 216.101 2257.94 221.888 M2253.68 220.638 Q2253.63 217.143 2251.71 215.059 Q2249.81 212.976 2246.67 212.976 Q2243.1 212.976 2240.95 214.99 Q2238.82 217.004 2238.49 220.661 L2253.68 220.638 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><circle clip-path="url(#clip100)" cx="1804.14" cy="270.476" r="20.48" fill="#e26f46" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="4.55111"/>
<path clip-path="url(#clip100)" d="M1910.86 264.816 Q1907.43 264.816 1905.44 267.501 Q1903.45 270.163 1903.45 274.816 Q1903.45 279.469 1905.42 282.154 Q1907.41 284.816 1910.86 284.816 Q1914.26 284.816 1916.25 282.131 Q1918.24 279.446 1918.24 274.816 Q1918.24 270.21 1916.25 267.524 Q1914.26 264.816 1910.86 264.816 M1910.86 261.205 Q1916.41 261.205 1919.58 264.816 Q1922.76 268.427 1922.76 274.816 Q1922.76 281.182 1919.58 284.816 Q1916.41 288.427 1910.86 288.427 Q1905.28 288.427 1902.11 284.816 Q1898.96 281.182 1898.96 274.816 Q1898.96 268.427 1902.11 264.816 Q1905.28 261.205 1910.86 261.205 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1948.43 274.816 Q1948.43 270.117 1946.48 267.455 Q1944.56 264.77 1941.18 264.77 Q1937.8 264.77 1935.86 267.455 Q1933.94 270.117 1933.94 274.816 Q1933.94 279.515 1935.86 282.2 Q1937.8 284.862 1941.18 284.862 Q1944.56 284.862 1946.48 282.2 Q1948.43 279.515 1948.43 274.816 M1933.94 265.765 Q1935.28 263.45 1937.32 262.339 Q1939.38 261.205 1942.22 261.205 Q1946.95 261.205 1949.89 264.955 Q1952.85 268.705 1952.85 274.816 Q1952.85 280.927 1949.89 284.677 Q1946.95 288.427 1942.22 288.427 Q1939.38 288.427 1937.32 287.316 Q1935.28 286.182 1933.94 283.867 L1933.94 287.756 L1929.65 287.756 L1929.65 251.737 L1933.94 251.737 L1933.94 265.765 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1976.44 262.594 L1976.44 266.622 Q1974.63 265.696 1972.69 265.233 Q1970.74 264.77 1968.66 264.77 Q1965.49 264.77 1963.89 265.742 Q1962.32 266.714 1962.32 268.659 Q1962.32 270.14 1963.45 270.997 Q1964.58 271.83 1968.01 272.594 L1969.47 272.918 Q1974.01 273.89 1975.9 275.672 Q1977.82 277.432 1977.82 280.603 Q1977.82 284.214 1974.95 286.321 Q1972.11 288.427 1967.11 288.427 Q1965.02 288.427 1962.76 288.01 Q1960.51 287.617 1958.01 286.807 L1958.01 282.409 Q1960.37 283.635 1962.66 284.26 Q1964.95 284.862 1967.2 284.862 Q1970.21 284.862 1971.83 283.844 Q1973.45 282.802 1973.45 280.927 Q1973.45 279.191 1972.27 278.265 Q1971.11 277.339 1967.15 276.483 L1965.67 276.135 Q1961.71 275.302 1959.95 273.589 Q1958.2 271.853 1958.2 268.844 Q1958.2 265.186 1960.79 263.196 Q1963.38 261.205 1968.15 261.205 Q1970.51 261.205 1972.59 261.552 Q1974.68 261.899 1976.44 262.594 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2006.78 273.728 L2006.78 275.811 L1987.2 275.811 Q1987.48 280.209 1989.84 282.524 Q1992.22 284.816 1996.46 284.816 Q1998.91 284.816 2001.2 284.214 Q2003.52 283.612 2005.79 282.409 L2005.79 286.436 Q2003.5 287.408 2001.09 287.918 Q1998.68 288.427 1996.2 288.427 Q1990 288.427 1986.37 284.816 Q1982.76 281.205 1982.76 275.047 Q1982.76 268.682 1986.18 264.955 Q1989.63 261.205 1995.46 261.205 Q2000.7 261.205 2003.73 264.585 Q2006.78 267.941 2006.78 273.728 M2002.52 272.478 Q2002.48 268.983 2000.56 266.899 Q1998.66 264.816 1995.51 264.816 Q1991.95 264.816 1989.79 266.83 Q1987.66 268.844 1987.34 272.501 L2002.52 272.478 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2028.8 265.811 Q2028.08 265.395 2027.22 265.21 Q2026.39 265.001 2025.37 265.001 Q2021.76 265.001 2019.82 267.362 Q2017.89 269.7 2017.89 274.098 L2017.89 287.756 L2013.61 287.756 L2013.61 261.83 L2017.89 261.83 L2017.89 265.858 Q2019.24 263.497 2021.39 262.362 Q2023.54 261.205 2026.62 261.205 Q2027.06 261.205 2027.59 261.274 Q2028.13 261.321 2028.77 261.436 L2028.8 265.811 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2030.21 261.83 L2034.72 261.83 L2042.82 283.589 L2050.93 261.83 L2055.44 261.83 L2045.72 287.756 L2039.93 287.756 L2030.21 261.83 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2073.1 274.723 Q2067.94 274.723 2065.95 275.904 Q2063.96 277.084 2063.96 279.932 Q2063.96 282.2 2065.44 283.543 Q2066.94 284.862 2069.51 284.862 Q2073.06 284.862 2075.19 282.362 Q2077.34 279.839 2077.34 275.672 L2077.34 274.723 L2073.1 274.723 M2081.6 272.964 L2081.6 287.756 L2077.34 287.756 L2077.34 283.821 Q2075.88 286.182 2073.7 287.316 Q2071.53 288.427 2068.38 288.427 Q2064.4 288.427 2062.04 286.205 Q2059.7 283.959 2059.7 280.209 Q2059.7 275.834 2062.62 273.612 Q2065.56 271.39 2071.37 271.39 L2077.34 271.39 L2077.34 270.973 Q2077.34 268.034 2075.39 266.436 Q2073.47 264.816 2069.98 264.816 Q2067.76 264.816 2065.65 265.348 Q2063.54 265.881 2061.6 266.946 L2061.6 263.01 Q2063.94 262.108 2066.13 261.668 Q2068.33 261.205 2070.42 261.205 Q2076.04 261.205 2078.82 264.122 Q2081.6 267.038 2081.6 272.964 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2094.58 254.469 L2094.58 261.83 L2103.36 261.83 L2103.36 265.14 L2094.58 265.14 L2094.58 279.214 Q2094.58 282.385 2095.44 283.288 Q2096.32 284.191 2098.98 284.191 L2103.36 284.191 L2103.36 287.756 L2098.98 287.756 Q2094.05 287.756 2092.18 285.927 Q2090.3 284.075 2090.3 279.214 L2090.3 265.14 L2087.18 265.14 L2087.18 261.83 L2090.3 261.83 L2090.3 254.469 L2094.58 254.469 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2108.96 261.83 L2113.22 261.83 L2113.22 287.756 L2108.96 287.756 L2108.96 261.83 M2108.96 251.737 L2113.22 251.737 L2113.22 257.131 L2108.96 257.131 L2108.96 251.737 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2132.18 264.816 Q2128.75 264.816 2126.76 267.501 Q2124.77 270.163 2124.77 274.816 Q2124.77 279.469 2126.74 282.154 Q2128.73 284.816 2132.18 284.816 Q2135.58 284.816 2137.57 282.131 Q2139.56 279.446 2139.56 274.816 Q2139.56 270.21 2137.57 267.524 Q2135.58 264.816 2132.18 264.816 M2132.18 261.205 Q2137.73 261.205 2140.9 264.816 Q2144.07 268.427 2144.07 274.816 Q2144.07 281.182 2140.9 284.816 Q2137.73 288.427 2132.18 288.427 Q2126.6 288.427 2123.43 284.816 Q2120.28 281.182 2120.28 274.816 Q2120.28 268.427 2123.43 264.816 Q2126.6 261.205 2132.18 261.205 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2172.69 272.108 L2172.69 287.756 L2168.43 287.756 L2168.43 272.247 Q2168.43 268.566 2166.99 266.737 Q2165.56 264.909 2162.69 264.909 Q2159.24 264.909 2157.25 267.108 Q2155.25 269.307 2155.25 273.103 L2155.25 287.756 L2150.97 287.756 L2150.97 261.83 L2155.25 261.83 L2155.25 265.858 Q2156.78 263.52 2158.84 262.362 Q2160.93 261.205 2163.63 261.205 Q2168.1 261.205 2170.39 263.983 Q2172.69 266.737 2172.69 272.108 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><circle clip-path="url(#clip100)" cx="1804.14" cy="322.316" r="20.48" fill="#3da44d" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="4.55111"/>
<path clip-path="url(#clip100)" d="M1912.36 326.563 Q1907.2 326.563 1905.21 327.744 Q1903.22 328.924 1903.22 331.772 Q1903.22 334.04 1904.7 335.383 Q1906.2 336.702 1908.77 336.702 Q1912.32 336.702 1914.45 334.202 Q1916.6 331.679 1916.6 327.512 L1916.6 326.563 L1912.36 326.563 M1920.86 324.804 L1920.86 339.596 L1916.6 339.596 L1916.6 335.661 Q1915.14 338.022 1912.96 339.156 Q1910.79 340.267 1907.64 340.267 Q1903.66 340.267 1901.3 338.045 Q1898.96 335.799 1898.96 332.049 Q1898.96 327.674 1901.88 325.452 Q1904.82 323.23 1910.63 323.23 L1916.6 323.23 L1916.6 322.813 Q1916.6 319.874 1914.65 318.276 Q1912.73 316.656 1909.24 316.656 Q1907.02 316.656 1904.91 317.188 Q1902.8 317.721 1900.86 318.786 L1900.86 314.85 Q1903.2 313.948 1905.39 313.508 Q1907.59 313.045 1909.68 313.045 Q1915.3 313.045 1918.08 315.962 Q1920.86 318.878 1920.86 324.804 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1948.29 314.665 L1948.29 318.647 Q1946.48 317.651 1944.65 317.165 Q1942.85 316.656 1941 316.656 Q1936.85 316.656 1934.56 319.295 Q1932.27 321.911 1932.27 326.656 Q1932.27 331.401 1934.56 334.04 Q1936.85 336.656 1941 336.656 Q1942.85 336.656 1944.65 336.17 Q1946.48 335.661 1948.29 334.665 L1948.29 338.6 Q1946.51 339.434 1944.58 339.85 Q1942.69 340.267 1940.53 340.267 Q1934.68 340.267 1931.23 336.586 Q1927.78 332.906 1927.78 326.656 Q1927.78 320.313 1931.25 316.679 Q1934.75 313.045 1940.81 313.045 Q1942.78 313.045 1944.65 313.462 Q1946.53 313.855 1948.29 314.665 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1959.91 306.309 L1959.91 313.67 L1968.68 313.67 L1968.68 316.98 L1959.91 316.98 L1959.91 331.054 Q1959.91 334.225 1960.76 335.128 Q1961.64 336.031 1964.31 336.031 L1968.68 336.031 L1968.68 339.596 L1964.31 339.596 Q1959.38 339.596 1957.5 337.767 Q1955.63 335.915 1955.63 331.054 L1955.63 316.98 L1952.5 316.98 L1952.5 313.67 L1955.63 313.67 L1955.63 306.309 L1959.91 306.309 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1974.28 313.67 L1978.54 313.67 L1978.54 339.596 L1974.28 339.596 L1974.28 313.67 M1974.28 303.577 L1978.54 303.577 L1978.54 308.971 L1974.28 308.971 L1974.28 303.577 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M1997.5 316.656 Q1994.07 316.656 1992.08 319.341 Q1990.09 322.003 1990.09 326.656 Q1990.09 331.309 1992.06 333.994 Q1994.05 336.656 1997.5 336.656 Q2000.9 336.656 2002.89 333.971 Q2004.89 331.286 2004.89 326.656 Q2004.89 322.05 2002.89 319.364 Q2000.9 316.656 1997.5 316.656 M1997.5 313.045 Q2003.06 313.045 2006.23 316.656 Q2009.4 320.267 2009.4 326.656 Q2009.4 333.022 2006.23 336.656 Q2003.06 340.267 1997.5 340.267 Q1991.92 340.267 1988.75 336.656 Q1985.6 333.022 1985.6 326.656 Q1985.6 320.267 1988.75 316.656 Q1991.92 313.045 1997.5 313.045 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip100)" d="M2038.01 323.948 L2038.01 339.596 L2033.75 339.596 L2033.75 324.087 Q2033.75 320.406 2032.32 318.577 Q2030.88 316.749 2028.01 316.749 Q2024.56 316.749 2022.57 318.948 Q2020.58 321.147 2020.58 324.943 L2020.58 339.596 L2016.3 339.596 L2016.3 313.67 L2020.58 313.67 L2020.58 317.698 Q2022.11 315.36 2024.17 314.202 Q2026.25 313.045 2028.96 313.045 Q2033.43 313.045 2035.72 315.823 Q2038.01 318.577 2038.01 323.948 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /></svg>

```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

