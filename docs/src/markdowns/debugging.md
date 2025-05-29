# Debugging ActionModels
ActionModels creates a Turing model under the hood. It is convenient for usability that users do not have to define their own Turing model, but debugging Turing models can be complex, even when they are hand-crafted.
This section provides some tips on how to debug ActionModels models, which can be useful when the model does not run as expected, and when it is unclear why.

## Test for errors in the action model itself
The first step whe debugging an action model is often to make sure that the action model function itself functions correctly.
The user should run the function step-by-step, replacing the parameters and states taken from the ModelAttributes object with values that are known to be correct.

## Test simulation
The next step is to ensure that the model works correctly for simulation.
It is an advantage to start here, because the complexity of Turing's inference engine can make it difficult to identify the source of errors.
The user should run the model with a standard set of parameters and observations, and check that the output matches the expected output.

## Check parameter inference
If the above steps work as expected, it is time to fit the model to data.
It is recommended to start with a small, handcrafted dataset, which is known to work with the model.
If this works, the user can return to the full dataset that the model originally errorred on.

## Check the data
Sometimes, an action model will error only with specific input series (given some parameters and an autodiff backend too).
This is often the case if the model fitting works with a subset of the data, but not with the full dataset.
In this case, it can be an advantage to fit the data with various subsets to find the specific session(s) that cause the problem.
Then, the user can repeat the steps above with only the problematic session(s) to see where the problem comes from. 

## Hand-code a minimal Turing model
If the above keep giving problems, users can create a hand-coded minimal Turing model with just the components needed to run the model. This might look like this, with a Rescorla-Wagner model:

```julia

@model function testmodel()
    
    #Prepare observations and actions
    observations = [0.1, 0.2, 0.3]
    actions = [0.1, 0.1, 0.4]

    #Sample parameters
    learning_rate ~ LogitNormal()
    action_noise ~ LogNormal()

    #Set initial value
    expected_value = 0.0

    #Go through each observation
    for (observation, action) in zip(observations, actions)

        #Update the expected value
        expected_value += learning_rate * (observation - expected_value)

        #Sample the action
        action ~ Normal(expected_value, action_noise)
    end
end

model = testmodel()

sample(model, NUTS, 1000)

```

If this function runs, there is some specificity in either the data or the population model specification - or, of course, and error in ActionModels - which is the problem. Users should try using the exact observations and actions that cause problems in the dataset, or to deliberately use very wide priors, and see if they can replicate the error. 

## Create an issue
If none of the above works, users should feel free to create an issue on the ActionModels Github repo, and we can do our best to help. Note that, if the error found above is known to be due to a specific autodifferentiation backend, the error should be reported to them - but it might be difficult to know exactly what caused the problem, and if the user is unable to create a proper MWE, we can perhaps help with that.


## Common sources of errors

### Check that bounded parameters are within bounds
One common source of problems is if bounded parameters are not within bounds. It is recommended to double-check that all parameters are bounded appropriately. This includes setting appropirate priors in the independent session population model, and using appropriate inverse link functions in the regression population model.

### Consider the choice of priors
Inappropriate priors can easily lead to plroblems with inference, either because they make the sampler not able to estimate the posterior, or simply because they lead to extreme parameter values that can cause errors.
It is especially important to consider if priors are well-specified when using the default priors in ActionModels, which may not be appropriate for the specific model.
Generally, it can be an advantage when debugging to start with very narrow priors to see if that allows the model to run, and then widen the priors from there.

### Consider underflow and overflow
Especially when sampling is it possible to get values that cannot be represented in the computer's memory.
This can easily happen when values get very close to some bound (e.g. 0 or 1 for probabilities). In this case, a value that should have been close to but not equal to 0 becomes 0, for example.
This can easily lead to infinities, NaN values and errors (for example if log-transformations are used).
If encountering such errors, it is recommended to check the values of the parameters and observations, and see if underflow is the culprit.
If it is, it can be a simple solution to add or subtract a simple epsilon value, or use `clamp` to ensure that values do not get too close to the bounds. Often, it is also a solution to use more narrow priors.

### Consider the choice of autodifferentiation backend
Turing supports multiple autodifferentiation backends, which have different advantages and disadvantages.
Some backends can lead to errors or wrong calculations in specific situations. 
To check whether a problem is related to the autodifferentiation backend, it is recommended to try running the model with a different backend.
The ForwardDiff backend is slower but more robust, so users may investigate whether switching to this backend solves the problem.
If it does solve the problem, the user may consider reporting the problem to the developers of the backend which gave the error, so that it can be fixed in future versions.
For this, it is often necessary to provide a minimal working example that reproduces the error, so that the developers can investigate it.
This can be a bit complicated to do, since it involves creating a Turing model from scratch, but if help is needed, the ActionModels developers can assist with this.
See the Turing [ADTest](https://turinglang.org/ADTests/) testsuite for examples of where the different backends might fail.

### Check that gradients can be calculated
It can sometimes be the case that gradients cannot be calculated for the parameters.
This can happen for various reasons, including the priors being ill specified. 
The following code command can be used to extract the gradient and logdensity, given a model, an autodiff backend, and some specified parameter values. It cna be convenient to use parameter values that are one sample from the prior, as below.

```julia

#Sample a single set of parameter values from the prior
params = DynamicPPL.VarInfo(model)[:]

#Calculate the logdensity and gradient
LogDensityProblems.logdensity_and_gradient(
    LogDensityFunction(model; adtype = ad_type),
    params,
)
```

### Matrix mutation
One point which can cause issues with some autodiff backends is matrix mutation, especially on multivariate states or parameters. The same operation can often be implemented without matrix mutation; it is recommended to test whether this can solve any problems.