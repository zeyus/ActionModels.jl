
## API ##
# TODO: (1.0) implement Regression input type
# TODO: (1.0) fix the errors in glm_tests.jl
# TODO: (1.0) example / usecase / tutorials)
#      - TODO: Fit a real dataset
# TODO: (1.0) implement check_population_model
#       - TODO: make check for whether there is a name collision with creating column with the parameter name
#       - TODO: check for whether the vector of priors is the correct amount
# TODO: (1.0) make a show or summarize function for the GLM chains object
#       - TODO: which renames parameters to be interpretable and removes the random effects etc
# TODO: allow using GLM with multivariate parameters - set an aray of regression models

## FUNCTIONALITY ##
# TODO: look at Turing's suggestions for regression models: https://turinglang.org/docs/tutorials/bayesian-linear-regression/
# TODO: use an mvnormal to sample all the random effects, instead of a huge constructed arraydist
# TODO: add covariance between random effects
# TODO: add option for independence between random effect groups (like brms' group_by = )
# TODO: check for type stability in the population model

## DOCUMENTATION ##
# TODO: add to documentation
#       - TODO: that there shouldn't be random slopes for the most specific level of session column (particularly when you only have one session column)


###################################
### REGRESSION POPULATION MODEL ###
###################################
function create_model(
    action_model::ActionModel,
    regressions::Union{F,Vector{F}},
    data::DataFrame;
    observation_cols::Union{
        NamedTuple{observation_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    action_cols::Union{
        NamedTuple{action_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    session_cols::Union{Vector{Symbol},Symbol} = Vector{Symbol}(),
    verbose::Bool = true,
    kwargs...,
) where {F<:Union{Regression,FormulaTerm},observation_names,action_names}

    ## Setup ##
    #If there is only one regression
    if regressions isa F
        #Put it in a vector
        regressions = F[regressions]
    end

    #Make sure that single formulas are made into Regression objects
    regressions = [
        regression isa Regression ? regression : Regression(regression) for
        regression in regressions
    ]

    #Check population_model
    check_population_model(
        RegressionPopulationModel(),
        action_model,
        regressions,
        data,
        observation_cols,
        action_cols,
        session_cols,
        verbose;
        kwargs...,
    )

    #Extract just the data needed for the linear regression
    population_data = unique(data, session_cols)
    #Extract number of sessions
    n_sessions = nrow(population_data)

    ## Condition single regression models ##
    #Initialize vector of sinlge regression models
    regression_models = Vector{DynamicPPL.Model}(undef, length(regressions))
    estimated_parameter_names = Vector{Symbol}(undef, length(regressions))

    #For each formula in the regression formulas, and its corresponding prior and link function
    for (model_idx, regression) in enumerate(regressions)

        #Extract information from the regression object
        formula = regression.formula
        prior = regression.prior
        inv_link = regression.inv_link

        #Prepare the data for the regression model
        X, Z = prepare_regression_data(formula, population_data)

        if has_ranef(formula)

            #Extract each function term (random effect part of formula)
            ranef_groups =
                [term for term in formula.rhs if term isa MixedModels.FunctionTerm]
            #For each random effect, extract the number of categories there are in the dataset
            n_ranef_categories = [
                nrow(unique(population_data, Symbol(term.args[2]))) for term in ranef_groups
            ]

            #Number of random effects
            size_r = size.(Z, 2)
            #For each random effect, extract the number of parameters
            n_ranef_params = [
                Int(size_rⱼ / n_ranef_categories[ranefⱼ]) for
                (ranefⱼ, size_rⱼ) in enumerate(size_r)
            ]

            #Full info for the random effect
            ranef_info = (
                Z = Z,
                n_ranef_categories = n_ranef_categories,
                n_ranef_params = n_ranef_params,
            )

            #Set priors
            internal_prior = RegPrior(
                β = if prior.β isa Vector
                    arraydist(prior.β)
                else
                    filldist(prior.β, size(X, 2))
                end,
                σ = if prior.σ isa Vector
                    arraydist.(prior.σ)
                else
                    [
                        filldist(prior.σ, Int(size(Zⱼ, 2) / n_ranef_categories[ranefⱼ])) for (ranefⱼ, Zⱼ) in enumerate(Z)
                    ]
                end,
            )
        else

            ranef_info = nothing

            #Set priors, and no random effects
            internal_prior = RegPrior(β = if prior.β isa Vector
                arraydist(prior.β)
            else
                filldist(prior.β, size(X, 2))
            end, σ = nothing)
        end

        #Condition the linear model
        regression_models[model_idx] =
            linear_model(X, ranef_info, inv_link = inv_link, prior = internal_prior)

        #Store the parameter name from the formula
        estimated_parameter_names[model_idx] = Symbol(formula.lhs)
    end

    #Create the combined regression population model
    population_model = regression_population_model(
        regression_models,
        estimated_parameter_names,
        n_sessions,
    )

    #Create a full model combining the session model and the population model
    return create_model(
        action_model,
        population_model,
        data;
        observation_cols = observation_cols,
        action_cols = action_cols,
        session_cols = session_cols,
        parameters_to_estimate = Tuple(estimated_parameter_names),
        population_model_type = RegressionPopulationModel(),
        kwargs...,
    )
end

## Turing model ##
@model function regression_population_model(
    linear_submodels::Vector{T},
    estimated_parameter_names::Vector{Symbol},
    n_sessions::Int,
) where {T<:DynamicPPL.Model}

    #Sample the parameters for each regression
    sampled_parameters = Tuple(
        p ~ to_submodel(prefix(linear_submodel, parameter_name), false) for
        (linear_submodel, parameter_name) in
        zip(linear_submodels, estimated_parameter_names)
    )

    return zip(sampled_parameters...)
end


"""
Generalized linear regression models following the equation:
η = X⋅β
with optionally:
for each random effect: η += Zⱼ⋅rⱼ
link function: link(η)
"""
@model function linear_model(
    X::Matrix{R1}, # model matrix for fixed effects
    ranef_info::Union{Nothing,T}; # model matrix for random effects
    inv_link::Function,
    prior::RegPrior,
    has_ranef::Bool = !isnothing(ranef_info),
) where {R1<:Real,T<:NamedTuple}

    #Sample beta / effect size parameters (including intercept)
    β ~ prior.β

    #Do fixed effect linear regression
    η = X * β

    #If there are random effects
    if has_ranef

        #Extract random effect information
        Z, n_ranef_categories, n_ranef_params = ranef_info

        #For each random effect
        for (ranefⱼ, (Zⱼ, n_ranef_categoriesⱼ, n_ranef_paramsⱼ)) in
            enumerate(zip(Z, n_ranef_categories, n_ranef_params))

            #Sample the individual random effects

            rⱼ ~ to_submodel(
                prefix(
                    sample_single_random_effect(
                        prior.σ[ranefⱼ],
                        n_ranef_categoriesⱼ,
                        n_ranef_paramsⱼ,
                    ),
                    "ranef_$ranefⱼ",
                ),
                false,
            )

            #Add the random effects to the linear model
            η += Zⱼ * rⱼ
        end
    end

    #Apply the link function, and return the resulting parameter for each participant
    return inv_link.(η)
end

## Sample a single random effect ##
@model function sample_single_random_effect(prior_σ, n_ranef_categoriesⱼ, n_ranef_paramsⱼ)

    #Sample random effect variance
    σ ~ prior_σ

    #Sample individual random effects
    r ~ arraydist([
        Normal(0, σ[param_idx]) for param_idx = 1:n_ranef_paramsⱼ for
        _ = 1:n_ranef_categoriesⱼ
    ])

    return r

end



###############################
####### HELPER FUNCTIONS ######
###############################
## Prepare the regression data structures ##
function prepare_regression_data(formula::FormulaTerm, population_data::DataFrame)
    #Inset column with the name fo the agetn parameter, to avoid error from MixedModel
    insertcols!(population_data, Symbol(formula.lhs) => 1) #TODO: FIND SOMETHING LESS HACKY

    if ActionModels.has_ranef(formula)
        X = MixedModel(formula, population_data).feterm.x
        Z = Matrix.(MixedModel(formula, population_data).reterms)
    else
        X = StatsModels.ModelMatrix(StatsModels.ModelFrame(formula, population_data)).m
        Z = nothing
    end

    return (X, Z)
end


## Check for random effects in a formula ##
function has_ranef(formula::FormulaTerm)

    #If there is only one term
    if formula.rhs isa AbstractTerm
        #Check if it is a random effect
        if formula.rhs isa FunctionTerm{typeof(|)}
            return true
        else
            return false
        end
        #If there are multiple terms
    elseif formula.rhs isa Tuple
        #Check if any are random effects
        return any(t -> t isa FunctionTerm{typeof(|)}, formula.rhs)
    end
end





##############################################
####### CHECKS FOR THE POPULATION MODEL ######
##############################################
function check_population_model(
    model_type::RegressionPopulationModel,
    action_model::ActionModel,
    regressions::Union{F,Vector{F}},
    data::DataFrame,
    observation_cols::Union{
        NamedTuple{observation_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    action_cols::Union{
        NamedTuple{action_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    session_cols::Union{Vector{Symbol},Symbol},
    verbose::Bool;
    kwargs...,
) where {F<:Regression,observation_names,action_names}

    #TODO: Make a check for whether there are NaN values in the predictors
    # if any(isnan.(data[!, predictor_cols]))
    #     throw(
    #         ArgumentError(
    #             "There are NaN values in the action columns",
    #         ),
    #     )
    # end

    #TODO: Make a check for whether the priors/linkfunctions/formulas are well-specified

end



########################################
####### DEFAULT PLOTTING FUNCTION ######
########################################
@recipe function f(modelfit::ModelFit{RegressionPopulationModel})

    throw(
        ArgumentError(
            "The regression population model does not yet have a plotting function",
        ),
    )

end
