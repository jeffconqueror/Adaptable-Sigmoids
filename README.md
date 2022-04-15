# Distance Classifier

We see many in other non-computational fields running classifiers that incorrectly classify something as a given class when in reality it is "none of the above." For Example, if we had a classifier that can classify cats and dogs, but is given a car. This classifier will incorrectly classify the car as either a dog or a cat when in reality, it is "None of the above."

There are no current algorithms that can accurately classify data as "none of the above", and output a rigorous confidence value. Neural Nets are specifically hard to get something rigorous form; if you look at the mathematics of how the final number pops out, it’s a weighted sum of edge weights. It would be difficult to get a number like 1e-9 out of a process like that, and The final output numbers typically are nowhere near that small. With this dilemma, we are developing an algorithm that can classify accurately as well as produce a rigorous P-value.

This is fork is a continuation of Tim's work


