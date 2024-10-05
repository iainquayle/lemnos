# problems

# changes

## guarenteed

### generation 

for dealing with the intercomponent shapes, and the current view, need to track the actual shape.
this could even be extended to pull in the prior shapes to enable lazy view changes.
to start though, it definitely appears that just tracking the current shape is necessary, 
ex, a flatten could happen prior to, or after the transformation, it isnt known until its actually evaluated.

UPDATE: in the end since python does not allow something like type classes, the best option is the map solution
to the best of my knowledge, with the binding solution, the class objects would need to prior to the schema creation have the functions bound,
which depending on, and to my best guess, of the cpython implementation would mean that they would need to be bound in the same file as the schema creation,
which wouldnt allow a framework agnostic system.
as well this de couples and part of the schema from the actual backend running and implementation.

something that may make everything easier is to make a factory pattern type statements system, our own very limit subsection of the ast.
make control of the users code much easier, and allow for better error messages and debugging.
shouldnt be too hard, but it seems like there is a better solution...
one thing is this could be fairly abstract, and be used across multiple backends and perhaps even languages if thats ever done.

### misc

shorten lines to 80 characters for everthing but templates perhaps

look into how the changes to ir will change the way in which breeding is done

fix layer norm, include rmsnorm

make biases optional

fix jank in sample collection

add vertical spacing to code, looks fine in nvim but vsc is disgusting

make tests for breeding

impl a screening on the ir, allow for culling on structural heuristics

work on making more controlable growth functions, ie, ones that would truely follow a given curve


## possible

rename anything to do with the schema compilation to solving, compilation is somewhat more of an adapter item, the schema is solved and the solution recorded  

allow an output conformance to be defined at compilation time, this may not be really possibe to do easily.

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
