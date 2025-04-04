
# TODO: (1.0) implement Regression input type
# TODO: implement check_population_model
#       - TODO: Make check for whether there is a name collision with creating column with the parameter name
#       - TODO: check for whether the vector of priors is the correct amount
# TODO: (1.0) Example / usecase / tutorials)
#      - TODO: Fit a real dataset
# TODO: add to documentation that there shouldn't be random slopes for the most specific level of grouping column (particularly when you only have one grouping column)
# TODO: add covariance between parameters
# TODO: Use an mvnormal to sample all the random effects, instead of a huge constructed arraydist
# TODO: make a show or summarize function for the GLM chains object
# TODO: fix the errors in glm_tests.jl

###################################
### REGRESSION POPULATION MODEL ###
###################################
struct RegressionPopulationModel <: AbstractPopulationModel end

function create_model(
    agent::Agent,
    regression_formulas::Union{F,Vector{F}},
    data::DataFrame;
    priors::Union{R,Vector{R}} = RegressionPrior(),
    inv_links::Union{Function,Vector{Function}} = identity,
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T2},
    grouping_cols::Union{Vector{T3},T3} = Vector{String}(),
    verbose::Bool = true,
    kwargs...,
) where {
    F<:MixedModels.FormulaTerm,
    R<:RegressionPrior,
    T1<:Union{String,Symbol},
    T2<:Union{String,Symbol},
    T3<:Union{String,Symbol},
}

    #Check population_model
    check_population_model(
        RegressionPopulationModel(),
        agent,
        regression_formulas,
        data,
        priors,
        inv_links,
        input_cols,
        action_cols,
        grouping_cols,
        verbose;
        kwargs...,
    )

    ## Setup ##
    #If there is only one formula
    if regression_formulas isa F
        #Put it in a vector
        regression_formulas = F[regression_formulas]
    end

    #If there is only one prior specified
    if priors isa RegressionPrior
        #Make a copy of it for each formula
        priors = RegressionPrior[priors for _ = 1:length(regression_formulas)]
    end

    #If there is only one link function specified
    if inv_links isa Function
        #Put it in a vector
        inv_links = Function[inv_links for _ = 1:length(regression_formulas)]
    end

    #Check that lengths are all the same
    if !(length(regression_formulas) == length(priors) == length(inv_links))
        throw(
            ArgumentError(
                "The number of regression formulas, priors, and link functions must be the same",
            ),
        )
    end

    #Extract just the data needed for the linear regression
    population_data = unique(data, grouping_cols)
    #Extract number of agents
    n_agents = nrow(population_data)

    ## Condition single regression models ##

    #Initialize vector of sinlge regression models
    regression_models = Vector{DynamicPPL.Model}(undef, length(regression_formulas))
    parameter_names = Vector{String}(undef, length(regression_formulas))

    #For each formula in the regression formulas, and its corresponding prior and link function
    for (model_idx, (formula, prior, inv_link)) in
        enumerate(zip(regression_formulas, priors, inv_links))

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
        parameter_names[model_idx] = string(formula.lhs)
    end

    #Create the combined regression statistical model
    population_model =
        regression_population_model(regression_models, parameter_names, n_agents)

    #Create a full model combining the agent model and the statistical model
    return create_model(
        agent,
        population_model,
        data;
        input_cols = input_cols,
        action_cols = action_cols,
        grouping_cols = grouping_cols,
        parameter_names = parameter_names,
        kwargs...,
    )
end

## Turing model ##
@model function regression_population_model(
    linear_submodels::Vector{T},
    parameter_names::Vector,
    n_agents::Int,
) where {T<:DynamicPPL.Model}

    #Sample the parameters for each regression
    sampled_parameters = Tuple(
        p ~ to_submodel(prefix(linear_submodel, parameter_name), false) for
        (linear_submodel, parameter_name) in zip(linear_submodels, parameter_names)
    )

    return revert(sampled_parameters)
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
function prepare_regression_data(
    formula::MixedModels.FormulaTerm,
    population_data::DataFrame,
)
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
    agent::Agent,
    regression_formulas::Union{F,Vector{F}},
    data::DataFrame,
    priors::Union{R,Vector{R}},
    inv_links::Union{Function,Vector{Function}},
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T2},
    grouping_cols::Union{Vector{T3},T3},
    verbose::Bool;
    kwargs...,
) where {
    F<:MixedModels.FormulaTerm,
    R<:RegressionPrior,
    T1<:Union{String,Symbol},
    T2<:Union{String,Symbol},
    T3<:Union{String,Symbol},
}

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







# function rename_chains(
#     chains::Chains,
#     model::DynamicPPL.Model,
#     #Arguments from population model
#     linear_submodels::Vector{T},
#     parameter_names::Vector,
#     n_agents::Int,
# ) where {T<:DynamicPPL.Model}


#     # Extract needed information

#     # For each regression

#     # Fixed effect names

#     # Random effect names

#     # replacement_names = Dict()
#     # for (param_name, _, __) in statistical_submodels
#     #     for (idx, id) in enumerate(eachrow(population_data[!,grouping_cols]))
#     #         if length(grouping_cols) > 1
#     #             name_string = string(param_name) * "[$(Tuple(id))]"
#     #         else
#     #             name_string = string(param_name) * "[$(String(id[first(grouping_cols)]))]"
#     #         end
#     #         replacement_names[string(param_name) * ".agent_param[$idx]"] = name_string
#     #     end
#     # end
#     # return replacenames(chains, replacement_names)
# end

