####################################
### RW STRUCT & UPDATE FUNCTIONS ###
####################################
## RW struct ##
abstract type AbstractRescorlaWagner <: AbstractSubmodel end

Base.@kwdef struct RescorlaWagner{T<:Union{Real,Array{Real}}} <: AbstractRescorlaWagner
    initial_value::Variable{T}
    learning_rate::Variable
end
## Continuous RW ##
function update!(attributes::RescorlaWagner{T}, observation::Float64) where {T<:Real}

    # Update the expected value using the Rescorla-Wagner rule
    attributes.expected_value =
        attributes.expected_value +
        attributes.learning_rate * (observation - attributes.expected_value)

    return nothing
end
## Binary RW ##
function update!(attributes::RescorlaWagner{T}, observation::Int64) where {T<:Real}

    #Get new value state
    attributes.expected_value +=
        attributes.learning_rate * (observation - logistic(attributes.expected_value))

    return nothing
end
## Categorical RW ##
function update!(
    attributes::CategoricalRescorlaWagner{T},
    observation::Int64,
) where {T<:Array{<:Real}}

    #Make one-hot encoded observation
    one_hot_observation = zeros(length(attributes.expected_value))
    one_hot_observation[observation] = 1

    # Update the expected value using the Rescorla-Wagner rule
    attributes.expected_value = map(
        (expected_value, observation) ->
            expected_value +=
                attributes.learning_rate * (observation - logistic(expected_value)),
        zip(attributes.expected_value, one_hot_observation),
    )

    return nothing
end




#############################
### INITIALIZE ATTRIBUTES ###
#############################
## Continuous and binary RW ##
function initialize_attributes(
    model::RescorlaWagner{T},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {TF,TI,T<:Real}

    #Initialize the attributes
    attributes =
        RescorlaWagner(initial_value = Variable{TF}(model.initial_value.value), learning_rate = Variable{TF}(model.learning_rate.value))

    #Set the expected value to the initial value
    attributes.expected_value = attributes.initial_value

    return attributes
end
## Categorical RW ##
function initialize_attributes(
    model::RescorlaWagner{T},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {TF,TI,T<:Array{<:Real}}

    #Initialize the attributes
    attributes = RescorlaWagner(
        initial_value = Variable{Array{TF}}(model.initial_value.value),
        learning_rate = Variable{TF}(model.learning_rate.value),
    )

    #Set the expected value to the initial value
    attributes.expected_value = zeros(length(attributes.initial_value))

    return attributes
end


##############################
### ATTRIBUTE MANIPULATION ###
##############################
## Reset function ##
function reset!(attributes::RescorlaWagner)
    # Reset the expected value to the initial value
    attributes.expected_value = attributes.initial_value
    return nothing
end

## Get single attribute ##
function get_parameters(attributes::RescorlaWagner, parameter_name::Symbol)
    if parameter_name in [:learning_rate, :initial_value]
        return attributes[parameter_name].value
    else
        return nothing #Let the higher level function handle the error
    end
end
function get_states(attributes::RescorlaWagner, state_name::Symbol)
    if state_name in [:expected_value]
        return attributes[state_name].value
    else
        return nothing #Let the higher level function handle the error
    end
end

## Get all attributes ##
function get_parameters(attributes::RescorlaWagner)
    return (;
        learning_rate = attributes.learning_rate.value,
        initial_value = attributes.initial_value.value,
    )
end
function get_states(attributes::RescorlaWagner)
    return (; expected_value = attributes.expected_value.value,)
end

## Set single attribute (used by attributes) ##
function set_parameters!(
    attributes::RescorlaWagner,
    parameter_name::Symbol,
    parameter_value::T,
) where {T<:Union{Real,Array{Real}}}
    if parameter_name in [:learning_rate, :initial_value]
        attributes[parameter_name].value = parameter_value
    else
        return false #Let the higher level function handle the error
    end
    return true
end
function set_states!(
    attributes::RescorlaWagner,
    state_name::Symbol,
    state_value::T,
) where {T<:Union{Real,Array{Real}}}
    if state_name in [:expected_value]
        attributes[state_name].value = state_value
    else
        return false #Let the higher level function handle the error
    end
    return true
end


## Set multiple attributes ##
function set_parameters!(
    attributes::RescorlaWagner,
    parameter_names::Tuple{Vararg{Symbol}},
    parameter_values::Tuple{Vararg{T}},
) where {T<:Union{Real,Array{Real}}}
    for (parameter_name, parameter_value) in zip(parameter_names, parameter_values)

        out = set_parameters!(attributes, parameter_name, parameter_value)

        #Raise an error if the parameter does not exist
        if !out
            error("Parameter $parameter_name does not exist in a Rescorla Wagner model.")
        end
    end
    return nothing
end
function set_states!(
    attributes::RescorlaWagner,
    state_names::Tuple{Vararg{Symbol}},
    state_values::Tuple{Vararg{T}},
) where {T<:Union{Real,Array{Real}}}
    for (state_name, state_value) in zip(state_names, state_values)

        out = set_states!(attributes, state_name, state_value)

        #Raise an error if the parameter does not exist
        if !out
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
                    rescorla_wagner::RescorlaWagner,
                    attributes::ModelAttributes,
                )
                    return Distributions.Normal(
                        rescorla_wagner.expected_value,
                        response_model_parameters.action_noise,
                    )
                end
                response_model_observations = (; observation = Observation(Float64))
                response_model_actions = (; report = Action(Normal))
            elseif type == :binary
                response_model = function bernoulli_report(
                    rescorla_wagner::RescorlaWagner,
                    attributes::ModelAttributes,
                )
                    return Distributions.Bernoulli(
                        logistic(
                            rescorla_wagner.expected_value *
                            response_model_parameters.action_noise,
                        ),
                    )
                end
                response_model_observations = (; observation = Observation(Int64))
                response_model_actions = (; report = Action(Bernoulli))
            elseif type == :categorical
                response_model = function categorical_report(
                    rescorla_wagner::RescorlaWagner,
                    attributes::ModelAttributes,
                )
                    return Distributions.Categorical(
                        softmax(
                            rescorla_wagner.expected_value *
                            response_model_parameters.action_noise,
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
        model_function = function rescorla_wagner(
            attributes::ModelAttributes{T},
            observation::R,
        ) where {T<:AbstractRescorlaWagner,R<:Real}

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
        model_function = function rescorla_wagner(
            attributes::ModelAttributes{T},
            observation::R,
        ) where {T<:AbstractRescorlaWagner,R<:Real}

            #Extract RW submodel
            rescorla_wagner = attributes.submodel

            #Update the Rescorla-Wagner expectation
            update!(rescorla_wagner, observation)

            #Run response model
            action_distribution = response_model(rescorla_wagner, attributes)

            return action_distribution
        end
    end

    ## Put model attriubtes in right format ##
    #Parameters
    parameters = NamedTuple(
        parameter_value isa AbstractFloat64 ?
        parameter_name => Parameter(parameter_value) :
        parameter_name => Parameter(parameter_value, discrete = true) for
        (parameter_name, parameter_value) in pairs(config.response_model_parameters)
    )
    #Observations
    observations = NamedTuple(
        observation_value isa AbstractFloat64 ?
        observation_name => Observation(observation_value) :
        observation_name => Observation(observation_value, discrete = true) for
        (observation_name, observation_value) in pairs(config.response_model_observations)
    )
    #Actions
    actions = NamedTuple(
        action_value isa AbstractFloat64 ?
        action_name => Action(action_value) :
        action_name => Action(action_value, discrete = true) for
        (action_name, action_value) in pairs(config.response_model_actions)
    )

    return ActionModel(
        model_function,
        parameters = parameters,
        observations = observations,
        actions = actions,
    )
end