############################################################
#### FUNCTION FOR RENAMING THE CHAINS OF A FITTED MODEL ####
############################################################
function rename_chains(chains::Chains, model::DynamicPPL.Model)
    #This will multiple dispatch on the type of statistical model
    rename_chains(chains, model, model.args.population_model.args...)
end

