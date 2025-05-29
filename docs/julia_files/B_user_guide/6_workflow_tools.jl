# # Tools for Bayesian workflows
# Although individual workdflows may vary depending on the requirements of the given project, there exists a range of standard methods that are recommended to be part of cognitive modelling workflows to ensure that analysis is completed correctly.
# These are introduced in more detail by various sources: see (Hess et al., 2025; Wilson & Collins, 2019; Lee & Wagenmakers, 2014) for examples.
# This includes prior and posterior checks, which ensure that the model produces appropriate behavioural, precision analysis and parameter recovery, which ensure that parameters can be accurately estimated from the data, and model comparison, which allows to compare models in terms of how well they describe the data.
# It also includes basic diagnostic checks, to ensure that the MCMC sampling has been succesful.
# ActionModels provides ready funtions to implement these methods, which are described in the following sections.

# ## Chain diagnostics
# ### Chain diagnostics with Turing
#TODO: diagnostics with Turing's own output
# This includes the rhat value, which indicates whether the chains have converged, and which should be close to 1 for all parameters.
# It also includes the `ess_bulk` and `ess_tail` values, which indicate the effective sample size of the chains.

# ### Chain diagnostics with ArviZ
#TODO: use ArviZ

# ### Looking for parameter correlations
#TODO: Make plot (or use ArviZ)

# ## Model comparison 
#TODO: use ArviZ

# ## Predictive checks 
#TODO: finish functions
# ### Prior predictive checks
# ### Posterior predictive checks

# ## Precision analysis 
#TODO: finish functions
# ### Parameter recovery
# ### Model recovery
# ### Experiment tuning
