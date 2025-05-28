# # Fitting action models to data

# In this section, we will cover how to fit actions models to data with ActionModels.jl.
# First, we will demonstrate how to set up the full model that will be fit using the Turing.jl framework.
# This consists of specifying an action model, a population model, and the data to fit the model to.
# Then, we will show how to sample from the posterior and prior distribution of the model parameters.
# And finally, we will show the tools in ActionModels to inspect and extract the results of the model fitting.

# ## Setting up the model
# First we import ActionModels, as well as StatsPlots for plotting the results later.
using ActionModels, StatsPlots

# We will here use the premade Rescorla-Wagner action model provided by ActionModels.jl. This is identical to the model described in the defining action models REF section.
action_model = ActionModel(RescorlaWagner())

# We will then specfiy the data that we want to fit the model to.
# For this example, we will use a simple manually created dataset, where three participants have completed an experiment where they must predict the next location of a moving target.
# Each participant has completed the experiment twice, in a control condition and under and experimental treatment.
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

# Finally, we will specify a population model, which is the model of how parameters vary between the different sessions in the data.
# There are various options when doing this, which are described in the population model REF section.
# Here, we will use a regression population model, where we assume that the learning rate and action noise parameters depend linearly on the experimental treatment.
# It is a hierarchical model, which means assuming that the parameters are sampled from a Gaussian distribution, where the mean of the distribution is a linear function of the treatment condition.
# This is specified with standard LMER syntax, and we use a logistic link function for the learning rate to ensure that it is between 0 and 1, and an exponential link function for the action noise to ensure that it is positive.
population_model = [
    Regression(@formula(learning_rate ~ treatment + (1 | id)), logistic),
    Regression(@formula(action_noise ~ treatment + (1 | id)), exp),
]

# Finally, we can combine the three components into a full model that can be fit to the data.
# This is done with the `create_model` function, which takes the action model, population model, and data as arguments.
# Additionally, we specify which columns in the data contain the actions, observations, and session identifiers.
# This creates an ActionModels.ModelFit object, which containts the full model, and will contain the sampling results after fitting.
model = create_model(
    action_model,
    population_model,
    data;
    action_cols = :actions,
    observation_cols = :observations,
    session_cols = [:id, :treatment],
)

# If there are multiple actions or observations, we can specify them as a NamedTuple mapping each action or observation to a column in the data.
# For example, if we had two actions and two observations, we could specify them as follows:

(action_name1 = :action_column_name1, action_name2 = :action_column_name2)

# The column names can also be specified as a vector of symbols, in which case it will be assumed that the order matches the order of actions or observations in the action model.

# Finally, there may be missing data in the dataset.
# If actions are missing, they can be imputed by ActionModels. This is done by setting the `impute_actions` argument to `true` in the `create_model` function.
# If `impute_actions` is not set to `true`, missing actions will simply be skipped during sampling instead.
# This is a problem for action models which depend on their previous actions.

# ## Fitting the model

# Now that the model is created, we are ready to fit it.
# This is done under the hood using MCMC sampling, which is provided by the Turing.jl framework.
# ActionModels provides the sample_posterior! function, which fits the model in this way with sensible defaults.

chns = sample_posterior!(model)

# This returns a MCMCChains Chains object, which contains the samples from the posterior distribution of the model parameters.
# The chains object contains metrics for whether the smapling was succesful. 
# This includes the rhat value, which indicates whether the chains have converged, and which should be close to 1 for all parameters.
# It also includes the ess_bulk and ess_tail values, which indicate the effective sample size of the chains.
# Finally, Turing provides plotting functions for the chains object, which can be used to visualize the results.
# We here deomstrate the basic plot function, which plots the traceplot and density of the posterior distributions of the parameters.
# See other plotting functions in the [MCMCChains documentation](https://turinglang.org/MCMCChains.jl/stable/statsplots/) for more options.
# Notably, not all plotting functions are suitable for plotting large amounts of parameters. 

plot(chns)

#TODO: ArviZ support


# Notably, sample_posterior! has many options for how to sample the posterior, which can be set with keyword arguments.
# If we pass either MCMCThreads() or MCMCDistributed() as the second argument, Turing will use multithreading or distributed sampling to parallellise between chains.
# It is recommended to use MCMCThreads for multithreading, but note that Julia must be started with the `--threads` flag to enable multithreading.
# We can specify the number of samples and chains to sample with the `n_samples` and `n_chains` keyword arguments.
# The init_params keyword argument can be used to specify how the initial parameters for the chains are set.
# It can either be set to :MAP or :MLE to use the maximum a posteriori or maximum likelihood estimates as the initial parameters, respectively, or to :sample_prior to draw a single sample from the prior distribution.
# If it is set to nothing, Turing's default of random values between -2 and 2 will be used as the initial parameters.
# Finally, arguments for the sampling can be passed. This includes the autodifferentiation backend to use, which can be set with the `ad_type` keyword argument, and the sampler to use, which can be set with the `sampler` keyword argument.
# Notably, sample_posterior! will return the already sampled Chains object if the posterior has already been sampled. Set `resample = true` to override the already sampled posterior.

chns = sample_posterior!(
    model,
    MCMCThreads(),
    n_samples = 500,
    n_chains = 4,
    init_params = :MAP,
    ad_type = AutoForwardDiff(),
    sampler = NUTS(),
    resample = true,
)

# ActionModels also provides functionality for saving segments of a chain and then resuming during sampling, so that long sampling runs can be interrupted and resumed later.
# This is done with passing a SampleSaveResume object to the `save_resume` keyword argument.
# The save_every keyword argument can be used to specify how often the chains should be saved to disk, and the path keyword argument specifies where the chains should be saved.
# Chains are saved with a prefix (by default "ActionModels_chain_segment") and a suffix that contains the chain and segment number.

ActionModels_path = dirname(dirname(pathof(ActionModels))) #hide
docs_path = joinpath(ActionModels_path, "docs") #hide

chns = sample_posterior!(
    model,
    save_resume = SampleSaveResume(
        path = joinpath(docs_path, ".samplingstate"),
        save_every = 200,
    ),
    n_samples = 600,
    resample = true,
)

# Finally, some users may wish to use Turing's own interface for sampling from the posterior instead.
# The Turing inferface is more flexible in general, but requires more boilerplate code to set up.
# For this case, the ActionModels.ModelFit objects contains the Turing model that is used under the hood. Users can extract and use it as any other Turing model, if they wish.

turing_model = model.model

# If users want to sample from the model themselves, but still want to draw on the rest of the ActionModels API, they can set it in the ModelFit object themselves by creating an ActionModels.ModelFitResult object.
# This should be passed to either the posterior or prior field of the ModelFit object, after which it will interface with the ActionModels API as normal.
using Turing
chns = sample(turing_model, NUTS(), 1000)

model.posterior = ActionModels.ModelFitResult(; chains = chns);


# In addition to sampling from the posterior, ActionModels also provides functionality for sampling from the prior distribution of the model parameters.
# This is done with the sample_prior! function, which works in a similar way to sample_posterior!.
# Notably, it is much simpler, due to not requiring a complex sampler. This means that it only takes n_chains and n_samples as keyword arguments.
prior_chns = sample_prior!(model, n_chains = 1, n_samples = 1000)


# ## Investigating the results
# ### Parameter per session
# ActionModels provides a set of tools for investigating the results of the model fitting.
# This primarily involves extracting the (prior or posterior) probability distribution over parameters for each session, and for the state trajectories they imply model.
# The session parameters can be extracted with the `get_session_parameters!` function, which returns a ActionModels.SessionParameters object.
# Whether to extract the session parameters from the posterior or prior distribution can be specified as the second argument.

session_parameters = get_session_parameters!(model)
prior_session_parameters = get_session_parameters!(model, :prior)

# ActioModels provides a convenient functionality for plotting the session parameters.
#TODO: plot(session_parameters)

# The ActionModels.SessionParameters object contains probability distributions for each parameter and for each session.
# These can be found in the value field, which contains nested NamedTuples and an AxisArray with the samples.

session_parameters.value.learning_rate[1]

# These can be plotted or summarized in various ways, depending on the users needs.

@show mean(session_parameters.value.learning_rate[1])
density(session_parameters.value.learning_rate[1])

# ActionModels provides a convenient function for summarizing the session parameters as a DataFrame.

median_df = summarize(session_parameters)

# This returns the median of each parameter for each session. 
# The user can pass other functions for summarizing the samples, as for example to calculate the standard deviation of the posterior.

std_df = summarize(session_parameters, std)

# This can be saved to disk or used for plotting in whichever way the user prefers.

# ### State trajectories per session
# Users can also extract the trajectory of states for each session with the `get_state_trajectories!` function, which returns an ActionModels.StateTrajectories object.
# These are often used to correlate with some external measure, such as neuroimaging data.
# The second argument specified which state to extract. Again, the user can also specify to have the prior state trajectories extracted by passing :prior as the second argument.
state_trajectories = get_state_trajectories!(model, :expected_value)

prior_state_trajectories = get_state_trajectories!(model, :expected_value, :prior)

# ActionModels provides functionality for plotting the state trajectories.
#TODO: plot(state_trajectories)

# The ActionModels.StateTrajectories object contains the prior or porsterior distribution over state trajectories for each session, and for each of the specified states.
# This is stored in the value field, which contains a NamedTuple with the state names and session names as keys, and an AxisArray with the samples across timesteps.

state_trajectories.value.expected_value[1] # The expectation state for the first session

# The third axis is the timestep axis, for which the probability distribution can be accessed:
@show mean(state_trajectories.value.expected_value[1][timestep=2]) #the mean of the expectation at the second timestep for the first session
density(state_trajectories.value.expected_value[1][timestep=2])

# This can also be summarized neatly in a DataFrame:

median_df = summarize(state_trajectories)

#And from here, it can be used for plotting or further analysis as desired by the user.
