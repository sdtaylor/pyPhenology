---
title: pyPhenology: A python framework for plant phenology modelling
tags:
    - plants, ecology, modeling, phenology
authors:
 - name: Shawn David Taylor
   Orcid: 0000-0002-6178-6903
   affiliation: 1
affiliations:
 - name: School of Natural Resources and Environment, University of Florida
   index: 1


date: 3 July 2018
bibliography: refs.bib
---

# Summary

Phenology, the timing of biological events, is an important aspect of environmental systems and phenology models have been in use since the mid 20th century [@chuin2017]. Phenology is a well established field of research and there are numerous model comparison studies which attempt to find the model which best explains a specific phenological event [@tang2016, @basler2016]. Many phenology models are well established [@chuine2013], yet studies frequently implement a custom codebase to evaluate them.  The pyPhenology package attempts to create a common modelling framework which can be used and referenced in phenology research. This will allow other researchers to check the implementation of mathematical models and add new models without rewriting core functionality. 

pyPhenology has an object oriented API where the same analysis code can be used regardless of the underlying model. The API is inspired by scikit-learn, having fit() and predict() methods for all models [@scikit-learn]. This allows for easy and reproducible code for phenology model selection and comparison studies. The package implements model fitting using built in optimizers from the scipy package, allowing end-users to fit models to their data with just a few lines of code [@jones2001]. A common phenology model requirement is setting certain parameters to fixed values, for example setting the first day for degree day accumulation to January 1. This is done via a simple argument in model initialization. Fitted models can be saved for later use or model parameters can exported as a python dictionary for inclusion other analysis. New models can be easily added and must only implement the actual model equations and a few checks of for adequate explanatory variables. All other requirements, such as fitting, predictions, or model saving, can be inherited from a parent class. 

pyPhenology was built with large scale analysis in mind and currently drives the continental scale phenology models on http://phenology.naturecast.org. A similar phenology modeling framework, phenor, is currently available for the R language [@hufkens2018]. 

# References
