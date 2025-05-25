[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

We follow the ColPrac guide for collaborative practices, and recommend new contributors to read through it.

There are three primary ways in which contributors may wish to improve ActionModels. They may wish to contribute to the growing library of premade models, to add features more generally, or to report and/or fix errors. All of them are deeply appreciated. If there is interest, do not hesitate to ask to be made a more permanent constributor to the package.

# Adding premade models
ActionModels contains a library of pre-created models that users can easily instantiate and use. These are fully modular from most of the source code, so users can contribute here needing very little except for understanding the general API of ActionModels.

Adding a new premade action model requires three steps:
1) creating a Julia file in the `src/premade_models` folder which implements the action model itself, and adding the path to that julia file in the main `src/ActionModels.jl` file.
2) adding a Julia file in the `test/testsuite/premade_models` which runs a comprehensive set of tests for the model
3) adding a Julia file in the `docs/julia_files/C_premade_models` folder which contains documentation for the new model

Users can see the PVL-Delta model which is already created as an illustrative example. If users want to create a premade model that relies on a submodel struct, they can instead see the Rescorla-Wagner model for an example. 

In the first step, the contributor should make sure to export the config struct used in the `ActionModel` constructor call. They should keep the model as flexible as possible, so that it can be used for many different datasets, should keep it as readable as possible, and should keep it type-stable.

In the second step, the contributor should make sure to test both simulating with the model and fitting it to data, and to test the full exported API of functions like `get_parameters` etc.

In the third step, the contributor should provide a brief theoretical (and perhaps formal) introduction to the model and its theoretical commitments, preferably with a reference to an article that describes it in more detail, and also clarify which types of tasks and data it is useful for. The user should describe all the variations of the model that is included, and provide examples of its use for each. Users may add a dataset to the `docs/example_datasets` folder that can be used with the model, if they find that it would be illustrative.

When the above steps are completed, a user can create a pull-request into the `dev` branch, and a maintainer of ActionModels will review the code before merging it.

# Adding features and fixing errors
Users are welcome to make an issue on the Github repository if they have encountered and error or unexpected behaviour, or if they have a suggestion for a new feature or an improvement to ActionModels.jl, so as to initialize discussions about the design of the implementation. They are also very welcome to create a pull-request with an update to the package to the `dev` branch, which will be reviewed by a maintainer of ActionModels before merging. In any case, we wish to keep the bar low for contribution, so potential contributors should not hesitate to get in touch.
