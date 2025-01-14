# problems

# changes

## guarenteed

### graph solver

move compilation and graph backend to c modules

### source gen

impl some kind of de dup for static members that are the same

### style

shorten lines to 80 characters for everthing but templates perhaps

### misc

clean samples naming, make members private

impl a screening on the ir, allow for culling on structural heuristics

the adapters should be split off into seperate packages. dont want to force the loading of torch if using jax.
in the same vein, examples should be split up. can include ones that only show ir generation but no use of adapters.
as well, somehow allow the installs to specify which version of torch to use? setup.py can take in arguments supposedly...

look at breeding again and if it can be less jank

make biases optional

fix jank in sample collection

make tests for breeding

work on making more controlable growth functions, ie, ones that would truely follow a given curve

## possible

consider taking out addressing by index in metrics, and require a float fraction
or, only allow access via the number of samples

rename anything to do with the schema compilation to solving, compilation is somewhat more of an adapter item, the schema is solved and the solution recorded  

allow none to be entered for shape bounds, then just inherit the dimensionality of the input.
(this may not work well as the systen allows for multiple different shapes to be passed in, so a user may expect 2d and get a 3d)

review whther more work can be done together with the conformance gathering and tracker mutation passes.
ie, caching the conformance gathering information and use it for tracker mutation if valid

may be the case that mutable priorities are not actually needed? since it pops the top of the stack in any case
check this on Unet example

add optional shape bounds to transitions, not to be conformed to but allow more flexibility in defining when to use transitions
may need to make "add group" take an explicit iterable instead of a arg list
it seems to always be possible to work around, but would be easier to not

move templates to jinja or the sort?
maybe just leave it as is unless moving to more verbose definitions, ie rolling custom kernels
