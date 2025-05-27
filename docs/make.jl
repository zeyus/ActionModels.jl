using ActionModels
using Documenter
using Literate
using Glob

## SET FOLDER NAMES ##
if haskey(ENV, "GITHUB_WORKSPACE")
    project_dir = ENV["GITHUB_WORKSPACE"]
else
    project_dir = pwd()
end
julia_files_folder = joinpath(project_dir, "docs", "julia_files")
markdown_files_folder = joinpath(project_dir, "docs", "src")
generated_files_folder = joinpath(markdown_files_folder, "generated")
theory_folder = joinpath(markdown_files_folder, "theory")


## GENERATE MARKDOWNS ##
#Remove old markdowns
for markdown_file in glob("*.md", generated_files_folder)
    rm(markdown_file)
end

#Create markdowns from julia files
for julia_file in glob("*/*.jl", julia_files_folder)

    Literate.markdown(
        julia_file,
        generated_files_folder,
        execute = true,
        documenter = true,
        #codefence = "```julia" => "```",
    )
end

#Including the index file 
Literate.markdown(
    joinpath(julia_files_folder, "E_others", "index.jl"),
    markdown_files_folder,
    execute = true,
    documenter = true,
)

#And the README
Literate.markdown(
    joinpath(julia_files_folder, "E_others", "README.jl"),
    project_dir,
    execute = true,
)



## GENERATE AND DEPLOY DOCS ##
DocMeta.setdocmeta!(
    ActionModels,
    :DocTestSetup,
    :(using ActionModels);
    recursive = true,
)

#Create documentation
makedocs(;
    modules = [ActionModels],
    authors = "Peter Thestrup Waade ptw@cas.au.dk, Christoph Mathys chmathys@cas.au.dk and contributors",
    #repo = "https://github.com/ComputationalPsychiatry/HierarchicalGaussianFiltering.jl/blob/{commit}{path}#{line}",
    sitename = "ActionModels.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ComputationalPsychiatry.github.io/ActionModels.jl",
        assets = String[],
        size_threshold = 1_500_000, ##MAKE THIS SMALLER?
    ),
    doctest = true,
    pages = [
        "Welcome to ActionModels" => [
            joinpath(".", "index.md"),
        ],
        "Theory" => [
            joinpath(".", "markdowns", "theory.md")
        ],
        "User Guide" => [
            joinpath(".", "generated", "2_defining_models.md"),
            joinpath(".", "generated", "3_simulation.md"),
            joinpath(".", "generated", "4_model_fitting.md"),
            joinpath(".", "generated", "5_population_models.md"),
            joinpath(".", "generated", "6_workflow_tools.md"),
            joinpath(".", "generated", "7_using_submodels.md"),
            joinpath(".", "markdowns", "debugging.md"),
        ],
        "Premade Models" =>[
            joinpath(".", "generated", "rescorla_wagner.md"),
            joinpath(".", "generated", "pvl_delta.md"),
        ],
        "Tutorials" => [
            joinpath(".", "generated", "example_jget.md"),
            joinpath(".", "generated", "example_igt.md"),
        ],
        "Full API" => [
            joinpath(".", "markdowns", "full_API.md"),
        ],
    ]
)

deploydocs(;
    repo = "github.com/ComputationalPsychiatry/ActionModels.jl",
    devbranch = "main",
)