# Lemnos 

## Overview

Lemnos is a library to aid in the discovery of efficient, performant and effective neural network architectures.

Rather than requiring the user to manually create and test models, or create bespoke search engines,
Model Make provides a generalized framework for defining search spaces and optimizing over them.

Check out these [examples](examples).

## Setup 

It is not on PyPi so will need to install from here.
Pip install from one of the adapters, only one available is the PyTorch adapter.

## Structure

The library has three major components, the schema, adapters, and control.

### Schema

Schemas express the search area using weighted directed graphs, which are walked in such a manner to create a working model. 
When built correctly, a schema can describe complex architectures, such as ResNet, or U-Net, or anything in between. 
The nodes bundle three operations often seen together in modern neural networks, the operation, the activation, and the normalization.
The edges represent the flow of data between nodes, with the weights describing how the graph is to be walked.

#### Limitations

Let it be known, that as of right now, schemas are limited to using modules that only take and produce a single tensor 
(ie, self-attention is fine but not regular attention).
As well, not all valid schemas are solvable by the current solver,
so if there is a shape inference that requires a longer lookahead than one, it will be pure luck if it is compiled.

#### Beware of...  

There is no debugger on why a schema is not compiling, so make them incrementally, and attempt to compile after each addition.
As well, the dimensionality of the shape bound for a node will define the dimensionality of the output, so None values can't be excluded.

### Adapters

Adapters are responsible for compiling the intermediate representation into a specific runnable backend, such as PyTorch,
and providing the necessary wrappers to run the models under the supervision of the control.
This backend agnostic approach allows the schema to be compiled to whatever backend is most appropriate 
and efficient for the user's needs without needing to change the schema or control.

### Control

The control manages the search process, monitoring and recording the performance of the models,
and deciding which models will be bred to create the next generation of models.
While this could for the moment be implemented within the schema, it is being kept seperate for for both clarity,
and possible future expansion into a distributed client-server model.
