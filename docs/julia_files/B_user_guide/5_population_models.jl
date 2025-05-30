# # Population models
# Apart from the action model, which describes how actions are chosen and states updated on a timestep-by-timestep basis, it is also necessary to specify a population model when fitting to data.
# The population model describes how the parameters of the action model are distributed across the sessions in the dataset (i.e. the population).
# ActionModels provides two types of pre-created population models that the user can choose from.
# One is the *independent sessions population model*, in which it is assumed that the parameters for each session are independent from each but other, but has the same prior.
# The other is the *linear regression population model*, in which it is assumed that the parameters of each session are related to external variables through a linear regression.
# The linear regression population model also allows for specifying a hierarchical structure (i.e. a random effect), which is often used to improve parameter estimation in cognitive modelling.
# Finally, users can also specify their own custom population model as a Turing model.
# If there is onle a single session in the dataset, only the independent sessions population model is appropriate, since there is no session structure to model.

# First, we load the ActionModels package, and StatsPlots for plotting results
using ActionModels
using StatsPlots

# We then specify the data that we want to our model to.
# For this example, we will use a simple manually created dataset, where three participants have completed an experiment where they must predict the next location of a moving target.
# Each participant has completed the experiment twice, in a control condition and under an experimental treatment.
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

# We specify which columns in the data correspond to the actions, observations, and session identifiers.
action_cols = :actions;
observation_cols = :observations;
session_cols = [:id, :treatment];

# Finally, we specify the action model. We here use the premade Rescorla-Wagner action model provided by ActionModels. This is identical to the model described in the defining action models REF section.
action_model = ActionModel(RescorlaWagner())


# ## Independent session population model
# To specify the independent session population model, we only need to specify the prior distribution for each parameter to estimate.
# This is specified as a `NamedTuple` with the parameter names as keys and the prior distributions as values.
# We here select a `LogitNormal` prior for the learning rate, since it is constrained to be between o and 1, and a `LogNormal` prior for the action noise, since it is constrained to be positive.

population_model = (learning_rate = LogitNormal(), action_noise = LogNormal());

# We can then create the full model, and sample from the posterior

model = create_model(
    action_model,
    population_model,
    data;
    action_cols = action_cols,
    observation_cols = observation_cols,
    session_cols = session_cols,
)

chains = sample_posterior!(model)

# We can see in the `Chains` object that each parameter is estimated separately for each session.
# This means that the population model parameters in this case are identical to the session parameters.
# Therefore, the standard plotting functions will plot the session parameters directly.

#TODO: plot(model)

# We see that the learning rate (by construction) is lower in the treatment condition, and that sessions A, B C have the lowest, middle and highest learning rates, respectively.

# We can also extract the session parameters as a `DataFrame` for further analysis.
parameter_df = summarize(get_session_parameters!(model))

show(parameter_df)

# ## Linear regression population models
# Often, the goal of cognitive modelling is to relate differences in parameter values to some external variables, such as treatment conditions or participant characteristics.
# This is often done with a linear regression, where point estimates of the parameters are predicted from the external variables.
# In this case, it is informationally advantageous to fit the linear model as part of the population model, rather than fitting the linear model separately and then using the point estimates as the population model.
# ActionModels provides a pre-made linear regression population model that can be used for this purpose.

# To specify a linear regression popluation model, we create a tuple of `Regression` objects, where each `Regression` object specifies a regression model for one of the parameters to estimate.
# The regression is specified with standard LMER syntax, where the formula is specified as a `@formula` object.
# Here, we predict each parameter from the treatment condition, and add a random intercept for each session ID (making this a classic hierarchical model).
# For each regression, we can also specify an inverse link function. This function transforms the output of the regression, and is commonly used to ensure that the resulting parameter values are in the correct range.
# Here, we use the `logistic` function for the learning rate (to ensure it is between 0 and 1) and the `exp` function for the action noise (to ensure it is positive).
# If no inverse link function is specified, the identity function is used by default.

population_model = [
    Regression(@formula(learning_rate ~ 1 + treatment + (1 | id)), logistic),
    Regression(@formula(action_noise ~ 1 + treatment + (1 | id)), exp),
];

# It is possible to specify the priors for the regression population model with the RegressionPrior constructor.
# It is possible to specify priors for the regression coefficients (β) and the standard deviation of the random effects (σ).
# If a single deistribution is specified for a given type, it is used for all parameters of that type.
# To set separate priors for each predictor or each random effect, a vector of distributions can be passed instead.
# Here, we specify a Student's t-distribution with 3 degrees of freedom for the regression coefficients (β) and an Exponential distribution with rate 1 for the standard deviation of the random effects (σ).
# These values are for use in the regression before the inverse link function is applied. This means that values above 5 are somewhat extreme, so these priors are probably too broad for most usecases.
prior = RegressionPrior(β = [TDist(3), Normal()], σ = Exponential(1))

population_model = [
    Regression(@formula(learning_rate ~ 1 + treatment + (1 | id)), logistic, prior),
    Regression(@formula(action_noise ~ 1 + treatment + (1 | id)), exp, prior),
]


plot(prior.β[1], label = "Intercept")
plot!(prior.β[2], label = "Effect of treatment")
plot!(prior.σ, label = "Random intercept std")
title!("Regression priors for the Rescorla-Wagner model")
xlabel!("Regression coefficient")
ylabel!("Density")

# We can then create the full model, and sample from the posterior
model = create_model(
    action_model,
    population_model,
    data;
    action_cols = action_cols,
    observation_cols = observation_cols,
    session_cols = session_cols,
)

chains = sample_posterior!(model)

# We can see that there are, for each parameter, two $\beta$ estimates, where the first is the intercept and the second is the treatment effect.
# We can also see that there is, for each random effect, a $\sigma$ estimate, which is the standard deviation of the random intercepts, as well as each of the sampled random intercepts for each session. 
# We can also see that the effect of the treatment is estimated to be negative for the learning rate, which is by construction.

title = plot(
    title = "Posterior over treatment effect",
    grid = false,
    showaxis = false,
    bottom_margin = -30Plots.px,
)
plot(
    title,
    density(chns[Symbol("learning_rate.β[2]")], title = "learning rate", label = nothing),
    density(chns[Symbol("action_noise.β[2]")], title = "action noise", label = nothing),
    layout = @layout([A{0.01h}; [B C]])
)

# When using the linear regression population model, the session parameters are not directly estimated, but rather the regression coefficients and random effects.
# We can still extract the session parameters as before, however.

parameter_df = summarize(get_session_parameters!(model))

show(parameter_df)

# ## Custom population models
# Finally, it is also possible to specify a custom population model for use with ActionModels.
# This is done by creating a conditioned Turing model that describes the population model, and which returns the sampled parameters for each session.
# The output of the model, which is used as a Turing submodel, must be iterable, and must for each session return a tuple with the parameter names as keys and the sampled values as values.
# The order of the sessions will be the same as the order of the sessions in the data, so it is important to ensure that the model returns the parameters in the correct order.
# Additionally, the names of the parameters is also passed as a Tuple to the model, so that the parameters can be correctly matched to the data. The order of the vector and the sampled parameters must match.
# Here, we create a custom population model where the learning rate and action noise for each session are sampled from multivariate normal distributions and then transformed.

#Load Turing
using Turing

#Get the number of sessions in the data
n_sessions = nrow(unique(data, session_cols))

#Create the Turing model for the custom population model
@model function custom_population_model(n_sessions::Int64)

    #Sample parameters for each session
    learning_rates ~ MvNormal(zeros(n_sessions), I)
    action_noises ~ MvNormal(zeros(n_sessions), I)

    #Transform the parameters to the correct range
    learning_rates = logistic.(learning_rates)
    action_noises = exp.(action_noises)

    #Return the parameters as an iterable of tuples
    return zip(learning_rates, action_noises)
end

#Condition the turing model
population_model = custom_population_model(n_sessions)

#Specify which parameters are estimated
parameters_to_estimate = (:learning_rate, :action_noise);

# We can then create the full model, and sample from the posterior
model = create_model(
    action_model,
    population_model,
    data;
    action_cols = action_cols,
    observation_cols = observation_cols,
    session_cols = session_cols,
    parameters_to_estimate = parameters_to_estimate,
)

chains = sample_posterior!(model)

# We can see that the parameter are estimated for each session, in non-transformed space.
# We can still extract the session parameters as before. This extracts the parameters in the version that they are passed to the action model - i.e., transformed.

parameter_df = summarize(get_session_parameters!(model))

show(parameter_df)


# ## Single session population model
# Notably, it is also possible to use ActionModels with only a single session.
# In this case, the actions and observations can be passed as two vectors, instead of passing a `DataFrame`.
# For single session models, only the independent session population model is appropriate, since there is no session structure to model.

#Create a single session dataset
observations = [1.0, 1, 1, 2, 2, 2]
actions = [0, 0.2, 0.3, 0.4, 0.5, 0.6]

#Create a standard independent sessions population model
population_model = (learning_rate = LogitNormal(), action_noise = LogNormal())

#Create full model
model = create_model(action_model, population_model, observations, actions)

#Sample from the posterior
chains = sample_posterior!(model)

# We can still plot the estimated parameters for the session
#TODO: plot(model)

