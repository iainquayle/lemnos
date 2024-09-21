# problems

# changes

## guarenteed

### generation 

two options, have a map that takes in the type and runs/returns the function, or,
overide a interface function that is blank to begin with to the class (not instance).

in the case of interface functions, there are two options.
have a single function that returns all statements, or have two functions, one for init and one for forward.
while jax flax only really needs the forward, it could be useful to have the init as well for constants.
however, they are fully jit compiled and likely would figure out the constants themselves.

unroll any sub module, and just run everythin in line. this allows for member reuse, 
ie in the instance that a module has two static tensors that are the exact same, then use them in both.
this can only be done for non callable members, use issubclass for this?

statements funcs will need, a identifier generator, scoped to each module but allow
these can be used in both the forward and the init statements, it may even be the case that they can be used for intermediate forward statements
will need to take in the the identifier of the input tensor, and perhaps the identifier of the output tensor
instead of passing in the output tensor, the output statement can be returned in a specific spot and be automatically assigned 
kind of seems overly complicated...
major problem with this idea is that one of the main necessities for pytorch is not at all need in jax flax impl, the identifier generator
other option is that there are multiple possible functions to choose from, and the assigner will choose which is correct,
and perhaps have a visible function that is called that will decide which is correct?
perhaps, as much as rt inrospection sucks, look at the inspect module
likely is far easier to just overwrite the functions, the main gain that is being looked for here is lsp friendly hints, but simply assigning functions is much easier than the alternatives

as for what they produce, the best options so far are a dict, or any and allow the user to decide

big problem with the generator binding, is that it would need to be done in the file, and prior to the schema definition as far as I can tell
this would kind of break the whole idea of keeping the schemas and the backend implementations separate

### misc

make biases optional

adapters, split up and fix any naming, and perhaps use a mapping instead of the if else chain
allow for the creation and use of multiple sub members of a node, then the need of custom module will be much reduced 

fix jank in sample collection

add vertical spacing to code, looks fine in nvim but vsc is disgusting

make tests for breeding

impl a screening on the ir, allow for culling on structural heuristics

work on making more controlable growth functions, ie, ones that would truely follow a given curve

fix layer norm

## possible

allow an output conformance to be defined at compilation time, this may not be really possibe to do easily.

transitions rename to edges perhaps

allow none to be entered for shape bounds, then just inherit the dimensionality of the input.
(this may not work well as the systen allows for multiple different shapes to be passed in, so a user may expect 2d and get a 3d)

review whther more work can be done together with the conformance gathering and tracker mutation passes.
ie, caching the conformance gathering information and use it for tracker mutation if valid

may be the case that mutable priorities are not actually needed? since it pops the top of the stack in any case
check this on Unet example

add optional shape bounds to transitions, not to be conformed to but allow more flexibility in defining when to use transitions
may need to make "add group" take an explicit iterable instead of a arg list
it seems to always be possible to work around, but would be easier to not

add a divisor option to the clamp val on shape bounds

move templates to jinja or the sort?
maybe just leave it as is unless moving to more verbose definitions, ie rolling custom kernels
