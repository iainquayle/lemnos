## problems

## changes
### guarenteed

making a new formatter, two options: make a map of functions and switch on the type, or,
make subclasses that inherit from both the parent component and a new backend specific formatter interface.
the first, while a little more janky is definitely easier, and really just as safe.
two sub options in that, when running map to the function calls, or,
do it ahead of time and have a overwritable function on the component and attach it to all instances.

make biases optional

adapters, split up and fix any naming, and perhaps use a mapping instead of the if else chain
allow for the creation and use of multiple sub members of a node, then the need of custom module will be much reduced 

fix jank in sample collection

add vertical spacing to code, looks fine in nvim but vsc is disgusting

make tests for breeding

impl a screening on the ir, allow for culling on structural heuristics

work on making more controlable growth functions, ie, ones that would truely follow a given curve

fix layer norm

### possible

allow an output conformance to be defined at compilation time, this may not be really possibe to do easily.

transitions rename to edges perhaps

allow none to be entered for shape bounds, then just inherit the dimensionality of the input.
may not be possible with how something like conv initialization works? will need to check

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
