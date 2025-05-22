# Computational cognitive modelling

Computational modelling of cognition and behaviour - or cognitive modelling for short - is a standard method in fields like mathematical psychology, computational psychiatry and model-based neuroscience.
The method consists of making formal and computational models of the mental processes that underlie observed behavior, in the same way computational models are made of other unobservable phenomena in the natural sciences.
The aim of cognitive modelling is provide mechanistic and formal accounts of mental processes. These can be used to test how well theories of cognitive function can describe empirically observed behaviour, and they can be used to make inference about individual differences in cognitive processes, for example between clinical populations.
Broadly speaking, cognitive models can be used in one of two ways. In theoretical cognitive modelling, cognitive processes and resulting behaviour is simulated in various environments and under various conditions to explore the implications of the models. In applied cognitive modeling, models are applied to empirical behavioural data (from human, animals, organoids or other real or simulated behavioural systems), in order to make inference about the processes that generated that behaviour.
Since both the fields of cognitive modelling and machine learning are in the business of designing computational models that generate behaviour, cognitive models can inspire as well as be inspired by more engineering-oriented models in machine learning fields, and questions, topics, methods and sometimes concrete models often overlap between the two fields. The primary difference is perhaps that machine learning purposes to a higher degree require computational and perfomance efficiency, while cognitive modelling require interpretability and theoretical plausibility as models of cognition.
There are various modeling approaches or paradigms within cognitive modeling, implying different theoretical commitments and used by different scientific communities. Some noteworthy paradigms might include cognitive architecture models CITE, Bayesian mind models CITE, vairational Bayesian predictive processing models like active inference CITE and hierarchical gaussian filters CITE, reinforcement learning models CITE, sequential sampling models CITE and deep learning models CITE, as well as the classic models in mathematical psychology that are created specifically to a given behavioural phenomenon, and are not necessarily committed to following a specific paradigm. Cognitive modelers also vary in their ontological interpretation of their models, ranging from realist mechanistic interpretations where cognitive models are taken to reflect real cognitive mechanisms at play, to pragmatic or behaviourist interpretations where cognitive models are taken as interpretable re-descriptions of data which are useful for highlighting specific aspects or behaviour, or investigating how behaviour varies across for example clinical populations.

## What is an action model?
<em>ActionModels.jl</em> is built conceptually on the action-perception loop, a ubiquitous metaphor in the cognitive and psychological sciences. Here, an agent (for example a human participant in an experiment) is conceptualized as receiving observations $o$ from an environment, and producing actions $a$. The environmental state $\varepsilon$ changes over time, partially dependent on the produced actions, so that the observations $o$ it produces can vary in complex and action-dependent ways. Environments can vary from a pre-defined set of observations (often used in cognitive and neuroscientific experiments) to more complex fully reactive environments. The agent's cognitive state $\vartheta$ also changes over time, partially dependent on observations $o$, and in turn produces actions $a$. Finally, the relation between $o$, $\vartheta$ and $a$ is governed by some cognitive parameters $\Theta$, which differ from the states $\vartheta$ in that parameters $\Theta$ do not within a given behavioural session. Note that $o$, $a$, $\vartheta$, and $\Theta$ can each be a set of multiple observations, actions, states and parameters, respectively.

![im_action_loop](../images/percact_loop_3.svg)

An <em>action model</em> $M_a$ is then a formal and computational model of how, at a single timeset $t$, cognitive states $\vartheta_t$ are updated and actions $a_t$ are generated, given some observation $o_t$, some previous cognitive states $\vartheta_{t-1}$ and the cognitive parameters $\Theta$. A classic example of an action model, which will be used throughout this tutorial, is the Rescorla-Wagner reinforcement learning model, with a Gaussian noise report action. Here, expectations $V$ about environmental outcomes are updated at each timestep based on observations $o$ and a learning rate $\alpha$:

$V_t = V_{t-1} + \alpha (o_t - V_{t-1})$

Actions are then sampled from a Gaussian distribution with the expected outcome $V_t$ as mean, and a noise parameter $\beta$ as standard deviation:

$a_t \sim \mathcal{N}(V_t, \beta)$

Here, parameters $\Theta$ consist of the learning rate $\alpha$, the action noise $\beta$, and the initial expectation $V_0$, and states $\vartheta_t$ consist of the expectation $V_t$. Other examples of parameters could be loss aversions or forgetting rates, and other examples of states could be Bayesian beliefs or prediction errors. 

Notably, it is common in the cognitive modelling literature to make a distinction between the perceptual model and the response model, where the perceptual model describes how the agent updates beliefs about the environment based on observations, and the response model describes how actions are subsequentially generated. In the Rescorla-Wagner model described above, the perceptual model would be the updating of $V_t$, and the response model would be the generation of $a_t$. The term <em>response model</em> comes from a conceptualization of actions as responses to stimuli; the response model is also sometimes called an <em>observation model</em>, to denote the decision-making process that underlies decision-based actions, or the <em>observation model</em>, because the agent's actions play the role of observations from the onlooking researcher's perspective. While a perception/action separation is appropriate for many contemporary models in the field, this is not necessarily always the case, so ActionModels does not impose this distinction. Instead, both components are subsumed under the term <em>action model</em>, which users may then structure as they would like.


## Theoretical cognitive modelling

Iris van Roij
Karl Friston

### Cognitive agent-based simulation

Agents.jl





## Applied cognitive modelling

In applied cognitive modelling, some dataset of empirically observed behaviour (including observation $o$ and actions $a$) from one of more behavioural systems has been obtained, and the goal is to make inference about the action model $M_a$, the cognitive parameters $\Theta$, and the ensuing cognitive states $\vartheta$ that generated the actions $a$, given the observations $o$.  

The behavioural systems which cognitive modelling can be applied to can in principle be any system that generates identifiable actions over some timespan, and usually given some observations. With the  models and model fitting techniques that currently exist, however, the method works best for relatively simple behavioural phenomena in structured environments. Most often, the method is used human participants participating in cognitive, psychological, psychiatric and neuroscientific experiments. It is also used with animal subjects, and can in principle be used with other biological systems like organoids and cells, as well as with artificial or simulated systems for which the behaviour-generating process is ill understood.

### Terminology in <em>ActionModels.jl</em>
An <em>action model</em> $M_a$ describes the evolution of states $\vartheta_t$ and generation of actions $a_t$ at a single timestep $t$, given some observation $o_t$, some previous cognitive states $\vartheta_{t-1}$ and the cognitive parameters $\Theta$. A set of timesteps $T$ over which the parameters $\Theta$ are constant is then called a session $s$, and the consecutive application of the action model across the timesteps of a session is called a <em>session model</em> $M_s$. A set of different sessions $S$ with varying parameters is then called a <em>population</em>, and the formal and computational model that describes how parameters $\Theta_s$ vary across sessions is called the <em>population model</em> $M_p$. In general, users of <em>ActionModels.jl</em> can specify the action model, the population model and the behavioural data - after this, the session models have been fully defined and are implemented automatically by the package.

![im_pop_model](../images/population_model.svg)

In the context of human psychological experiments, a sessionusually denotes a single experimental task, where the timesteps correspond to each trial in the task. In many experiments, each participant completes a single task, making each session $s$ also correspond to each participant, but in the more general case there may be multiple sessions belonging to a single participant.  

### Population models




Often, parameters are estimated separately between sessions in an experiment, which corresponds to a population model where sessions are independent; other population models include hierarchical models, where session parameters are taken to be sampled from some population-level distribution, or linear regression models, where session parameters are taken to depend on some external predictors. 
- population models:
- hierarchical parameter fitting
- include linear regression
- others


### Use cases
#### Comparing models
#### Computational phenotyping
- computational phenotyping / how parameters vary
#### Relate states to other things
- relating states to other things
- neuroimaging 
- physiology
- subjective ratings (e.g. confidences)
- reaction times
#### Technicalities of model fitting
- Fitting methods
- MCMC, Gibss vs HMC, autodiff packages
- alternatives: variational methods, optimization etc



## References
- One and Many
- Michael Lee Book
- Iris Book
- HBayesDM
- brms
- Agents.jl
- bayesian mind models
- active inference
- hgf
- deep learning as cognitive model
- sequential sampling
- cognitive architecture


