####################################
### RW STRUCT & UPDATE FUNCTIONS ###
####################################
## RW structs ##
abstract type AbstractRescorlaWagner <: AbstractSubmodel end

#Model struct
Base.@kwdef struct CategoricalRescorlaWagner <: AbstractRescorlaWagner
    initial_value::Vector{Float64}
    learning_rate::Float64
end
Base.@kwdef struct RescorlaWagner <: AbstractRescorlaWagner
    initial_value::Float64
    learning_rate::Float64
end

#Attributes struct
Base.@kwdef mutable struct RescorlaWagnerAttributes{T<:Real,AT<:Union{T,Array{T}}} <:
                           AbstractRescorlaWagner
    initial_value::AT
    learning_rate::T
    expected_value::AT
end
## Continuous RW ##
function update!(
    attributes::RescorlaWagnerAttributes{T,AT},
    observation::Float64,
) where {T<:Real,AT<:Real}
    # Update the expected value using the Rescorla-Wagner rule
    attributes.expected_value +=
        attributes.learning_rate * (observation - attributes.expected_value)
end
## Binary RW ##
function update!(
    attributes::RescorlaWagnerAttributes{T,AT},
    observation::Int64,
) where {T<:Real,AT<:Real}
    #Get new value state
    attributes.expected_value +=
        attributes.learning_rate *
        (observation - logistic(attributes.expected_value))
end
## Categorical RW ##
function update!(
    attributes::RescorlaWagnerAttributes{T, AT},
    observation::Int64,
) where {T<:Real,AT<:Array{T}}
    #Make one-hot encoded observation
    one_hot_observation = zeros(length(attributes.expected_value))
    one_hot_observation[observation] = 1

    update!(attributes, one_hot_observation)
end
## Categorical RW with binary vector input ##
function update!(
    attributes::RescorlaWagnerAttributes{T, AT},
    observation::Vector{Int64},
) where {T<:Real,AT<:Array{T}}
    attributes.expected_value = map(
        (expected_value, single_observation) ->
            expected_value +=
                attributes.learning_rate *
                (single_observation - logistic(expected_value)),
        zip(attributes.expected_value, observation),
    )
end
## Categorical RW with continuous vector input ##
function update!(
    attributes::RescorlaWagnerAttributes{T, AT},
    observation::Vector{Float64},
) where {T<:Real,AT<:Array{T}}
    attributes.expected_value = map(
        (expected_value, single_observation) ->
            expected_value +=
                attributes.learning_rate *
                (single_observation - expected_value),
        zip(attributes.expected_value, observation),
    )
end




#############################
### INITIALIZE ATTRIBUTES ###
#############################
## Continuous and binary RW ##
function initialize_attributes(
    model::RescorlaWagner,
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {TF,TI}

    #Initialize the attributes
    attributes = RescorlaWagnerAttributes{TF, TF}(
        initial_value = model.initial_value,
        expected_value = model.initial_value,
        learning_rate = model.learning_rate,
    )

    return attributes
end
## Categorical RW ##
function initialize_attributes(
    model::CategoricalRescorlaWagner,
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {TF,TI}

    #Initialize the attributes
    attributes = RescorlaWagnerAttributes{TF, Array{TF}}(
        initial_value = model.initial_value,
        expected_value = model.initial_value,
        learning_rate = model.learning_rate,
    )

    return attributes
end

## Functions for getting the types of the parameters and states ##
function get_parameter_types(model::RescorlaWagner)
    return (initial_value = Float64, learning_rate = Float64)
end
function get_state_types(model::RescorlaWagner)
    return (; expected_value = Float64,)
end
function get_parameter_types(model::CategoricalRescorlaWagner)
    return (initial_value = Array{Float64}, learning_rate = Float64)
end
function get_state_types(model::CategoricalRescorlaWagner)
    return (expected_value = Array{Float64},)
end

##############################
### ATTRIBUTE MANIPULATION ###
##############################
## Reset function ##
function reset!(attributes::RescorlaWagnerAttributes)
    # Reset the expected value to the initial value
    attributes.expected_value = attributes.initial_value
    return nothing
end

## Get single attribute ##
function get_parameters(attributes::RescorlaWagnerAttributes, parameter_name::Symbol)
    if parameter_name in [:learning_rate, :initial_value]
        return getfield(attributes, parameter_name)
    else
        return AttributeError() #Let the higher level function handle the error
    end
end
function get_states(attributes::RescorlaWagnerAttributes, state_name::Symbol)
    if state_name in [:expected_value]
        return getfield(attributes, state_name)
    else
        return AttributeError() #Let the higher level function handle the error
    end
end

## Get all attributes ##
function get_parameters(attributes::RescorlaWagnerAttributes)
    return (;
        learning_rate = attributes.learning_rate,
        initial_value = attributes.initial_value,
    )
end
function get_states(attributes::RescorlaWagnerAttributes)
    return (; expected_value = attributes.expected_value,)
end

## Set single attribute (used by attributes) ##
function set_parameters!(
    attributes::RescorlaWagnerAttributes,
    parameter_name::Symbol,
    parameter_value::T,
) where {T<:Union{Real,Array{Real}}}
    if parameter_name in [:learning_rate, :initial_value]
        setfield!(attributes, parameter_name, parameter_value)
    else
        return AttributeError() #Let the higher level function handle the error
    end
    return true
end
function set_states!(
    attributes::RescorlaWagnerAttributes,
    state_name::Symbol,
    state_value::T,
) where {T<:Union{Real,Array{Real}}}
    if state_name in [:expected_value]
        setfield!(attributes, state_name, state_value)
    else
        return AttributeError() #Let the higher level function handle the error
    end
    return true
end


## Set multiple attributes ##
function set_parameters!(
    attributes::RescorlaWagnerAttributes,
    parameter_names::Tuple{Vararg{Symbol}},
    parameter_values::Tuple{Vararg{T}},
) where {T<:Union{Real,Array{Real}}}
    for (parameter_name, parameter_value) in zip(parameter_names, parameter_values)

        out = set_parameters!(attributes, parameter_name, parameter_value)

        #Raise an error if the parameter does not exist
        if out isa AttributeError
            error("Parameter $parameter_name does not exist in a Rescorla Wagner model.")
        end
    end
    return nothing
end
function set_states!(
    attributes::RescorlaWagnerAttributes,
    state_names::Tuple{Vararg{Symbol}},
    state_values::Tuple{Vararg{T}},
) where {T<:Union{Real,Array{Real}}}
    for (state_name, state_value) in zip(state_names, state_values)

        out = set_states!(attributes, state_name, state_value)

        #Raise an error if the parameter does not exist
        if out isa AttributeError
            error("State $state_name does not exist in a Rescorla Wagner model.")
        end
    end
    return nothing
end







###################
### CONFIG TYPE ###
###################
export PremadeRescorlaWagner

struct PremadeRescorlaWagner <: AbstractPremadeModel
    #RW preceptual model attributes
    type::Symbol
    initial_value::Union{Float64,Vector{Float64}}
    learning_rate::Float64

    #Response model attributes
    response_model::Function
    response_model_parameters::NamedTuple{
        parameter_names,
        <:Tuple{Vararg{Parameter}},
    } where {parameter_names}
    response_model_observations::Union{
        Nothing,
        NamedTuple{observation_names,<:Tuple{Vararg{Observation}}},
    } where {observation_names}
    response_model_actions::NamedTuple{
        action_names,
        <:Tuple{Vararg{Action}},
    } where {action_names}
    act_before_update::Bool

    function PremadeRescorlaWagner(;
        type::Symbol = :continuous,
        initial_value::Union{Float64,Vector{Float64}} = 0.0,
        learning_rate::Float64 = 0.1,

        response_model::Union{Nothing,Function} = nothing,
        response_model_parameters::Union{
            Nothing,
            NamedTuple{parameter_names,<:Tuple{Vararg{Parameter}}},
        } where {parameter_names} = nothing,
        response_model_observations::Union{
            Nothing,
            NamedTuple{observation_names,<:Tuple{Vararg{Observation}}},
        } where {observation_names} = nothing,
        response_model_actions::Union{
            Nothing,
            NamedTuple{action_names,<:Tuple{Vararg{Action}}},
        } where {action_names} = nothing,
        action_noise::Union{Float64,Nothing} = nothing,
        act_before_update::Bool = false,
    )
        #Check if the type is valid
        if !(type in [:continuous, :binary, :categorical])
            error("Type $type is not a valid Rescorla-Wagner type.")
        end

        #Check if the type mathes the initial value
        if type == :categorical && initial_value isa Float64
            error(
                "Categorical Rescorla-Wagner models must have a vector of Float64 as initial value.",
            )
        end
        if type in [:continuous, :binary] && initial_value isa Vector{Float64}
            error(
                "Continuous and binary Rescorla-Wagner models must have a Float64 as initial value.",
            )
        end

        #If a default response model should be used
        if isnothing(response_model)

            ## Check that attributes have not been set ##
            if !isnothing(response_model_parameters)
                #Disallow setting the response model parameters keyword argument if a custom response model is not provided
                error(
                    "A custom response model has not been provided. Set the action noise with the action_noise keyword argument.",
                )
            end
            if !isnothing(response_model_observations)
                #Disallow setting the response model observations keyword argument if a custom response model is not provided
                error(
                    "A custom response model has not been provided. Set the observations with the response_model_observations keyword argument.",
                )
            end
            if !isnothing(response_model_actions)
                #Disallow setting the response model actions keyword argument if a custom response model is not provided
                error(
                    "A custom response model has not been provided. Set the actions with the response_model_actions keyword argument.",
                )
            end

            #If action noise is not set
            if isnothing(action_noise)
                #Set it to the default value
                action_noise = 1.0
            end

            #Set the response model parameters
            response_model_parameters = (; action_noise = Parameter(action_noise))

            #Set the response model
            if type == :continuous
                response_model = function gaussian_report(
                    rescorla_wagner::RescorlaWagnerAttributes,
                    attributes::ModelAttributes,
                )
                    return Distributions.Normal(
                        rescorla_wagner.expected_value,
                        load_parameters(attributes).action_noise,
                    )
                end
                response_model_observations = (; observation = Observation(Float64))
                response_model_actions = (; report = Action(Normal))
            elseif type == :binary
                response_model = function bernoulli_report(
                    rescorla_wagner::RescorlaWagnerAttributes,
                    attributes::ModelAttributes,
                )
                    return Distributions.Bernoulli(
                        logistic(
                            rescorla_wagner.expected_value *
                            load_parameters(attributes).action_noise,
                        ),
                    )
                end
                response_model_observations = (; observation = Observation(Int64))
                response_model_actions = (; report = Action(Bernoulli))
            elseif type == :categorical
                response_model = function categorical_report(
                    rescorla_wagner::RescorlaWagnerAttributes,
                    attributes::ModelAttributes,
                )
                    return Distributions.Categorical(
                        softmax(
                            rescorla_wagner.expected_value *
                            load_parameters(attributes).action_noise,
                        ),
                    )
                end
                response_model_observations = (; observation = Observation(Int64))
                response_model_actions = (; report = Action(Categorical))
            end
        else
            #Disallow setting the action noise keyword argument if a custom response model is provided
            if !isnothing(action_noise)
                error(
                    "A custom response model has been provided. Set it's parameters with the response_model_parameters argument.",
                )
            end
            #Disallow setting the response model observations keyword argument if a custom response model is provided
            if !isnothing(response_model_observations)
                #Disallow setting the response model observations keyword argument if a custom response model is provided
                error(
                    "A custom response model has been provided. Set it's observations with the response_model_observations argument.",
                )
            end
            #Disallow setting the response model actions keyword argument if a custom response model is provided
            if !isnothing(response_model_actions)
                #Disallow setting the response model actions keyword argument if a custom response model is provided
                error(
                    "A custom response model has been provided. Set it's actions with the response_model_actions argument.",
                )
            end
        end

        return new(
            type,
            initial_value,
            learning_rate,
            response_model,
            response_model_parameters,
            response_model_observations,
            response_model_actions,
            act_before_update,
        )
    end
end



########################################
### FUNCTION FOR GENERATING SUBMODEL ###
########################################
function ActionModel(config::PremadeRescorlaWagner)

    #Extract response model
    response_model = config.response_model

    ## - Create action model function - ##
    if config.act_before_update

        ## With action selection before expectation update ##
        model_function = function rescorla_wagner_act_before_update(
            attributes::ModelAttributes,
            observation::R,
        ) where {R<:Union{Real,Array{<:Real}}}

            #Extract RW submodel
            rescorla_wagner = attributes.submodel

            #Run response model
            action_distribution = response_model(rescorla_wagner, attributes)

            #Update the Rescorla-Wagner expectation
            update!(rescorla_wagner, observation)

            return action_distribution
        end

    else
        ## With expectation update before action selection ##
        model_function = function rescorla_wagner_act_after_update(
            attributes::ModelAttributes,
            observation::R,
        ) where {R<:Union{Real,Array{<:Real}}}

            #Extract RW submodel
            rescorla_wagner = attributes.submodel

            #Update the Rescorla-Wagner expectation
            update!(rescorla_wagner, observation)

            #Run response model
            action_distribution = response_model(rescorla_wagner, attributes)

            return action_distribution
        end
    end

    ## Generate RW Submodel ##
    if config.type == :categorical
        #Create the submodel
        RW_submodel = CategoricalRescorlaWagner(
            initial_value = config.initial_value,
            learning_rate = config.learning_rate,
        )
    else
        #Create the submodel
        RW_submodel = RescorlaWagner(
            initial_value = config.initial_value,
            learning_rate = config.learning_rate,
        )
    end

    return ActionModel(
        model_function,
        parameters = config.response_model_parameters,
        observations = config.response_model_observations,
        actions = config.response_model_actions,
        submodel = RW_submodel,
    )
end