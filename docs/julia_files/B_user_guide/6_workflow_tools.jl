# # Addtional tools for Bayesian analysis workflows
# Although individual workdflows may vary depending on the requirements of the given project, there exists a range of standard methods that are recommended to be part of cognitive modelling workflows to ensure that analysis is completed correctly.
# These are often introduced in more detail by various sources: see Hess et al., 2025, Wilson & Collins, 2019 or Lee & Wagenmakers, 2014 for examples.
# This includes prior and posterior checks, which ensure that the model produces appropriate behavioural, precision analysis and parameter recovery, which ensure that parameters can be accurately estimated from the data, and model comparison, which allows to comin terms of how well they describe the data.
# ActionModels provides ready funtions to implement these methods, which are described in the following sections.

# ## Chain diagnostics
#TODO: diagnostics with Turing's own output
# This includes the rhat value, which indicates whether the chains have converged, and which should be close to 1 for all parameters.
# It also includes the `ess_bulk` and `ess_tail` values, which indicate the effective sample size of the chains.
#TODO: use ArviZ
#TODO: parameter correlations

# ## Parameter correlations
#TODO: make plot function (or use ArviZ)

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
