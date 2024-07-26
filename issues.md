## problems

it should be possible to input a output conformance to the schema for comp, have found issue with it

## changes
### guarenteed

transitions rename to edges perhaps

adapters, split up and fix any naming, and perhaps use a mapping instead of the if else chain
allow for the creation and use of multiple sub members of a node, then the need of custom module will be much reduced 

fix jank in sample collection

add vertical spacing to code, looks fine in nvim but vsc is disgusting

make tests for breeding

impl a screening on the ir, allow for culling on structural heuristics

work on making more controlable growth functions, ie, ones that would truely follow a given curve

fix layer norm

### possible

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
