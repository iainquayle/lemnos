# Model Make

## Overview

Model Make is a library for optimizing machine learning models on a structural level. 

Model make 

## Setup 

... 

## Structure

The library has three major components, the schema, adapter, and control.
The search area is defined via user-defined graphs called schemas, from which models are generated.
The schemas produce an intermediate representation, which is then compiled by an adapter to a specific backend.
While model make comes with a default adapter for PyTorch, it is designed to be easily extended to other backends.
Finally, the control component optimizes via or-search(genetic algorithm). 

### Schema

The 
