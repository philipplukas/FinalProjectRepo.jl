# Part 2: Shallow Water Equations
For the second part, I chose to write a XPU solver for the Shallow Water Equations.

## Intro
The sallow water quations can be derived from the Navier Stokes equations. The main difference is, that we don't care about the z-direction. Rather, we consider the average of the z-direction in terms of speed in x/y directions. This process is called depth integration. This simplyfing assumption requires that the horizontal dimesnion is much larger than the vertical one. I implemented the conservative form of those euqations, which ignore stress, viscuous and corriolis forces. I implemented the following equations from Wikpedia.

<img width="350" alt="Screenshot 2021-12-20 at 14 21 34" src="https://user-images.githubusercontent.com/18243049/146821128-af4df50c-2dd5-49de-8e08-18d0f4c5826f.png">

Then, I also used my code to model the dam-break problem. I have two versions of the dam-break problem code. Once, I used a simple implementation, running on the CPU. For the other implementation, I used parallel_stencil to parallelize the code on a GPU.


## Methods

I tried to implement the equations on Wikpedia directly using the forward method. However, this method is very unstable.
Therefore, I used the Lax-Friedrich method to have better propoerties of the numeriadl behavior.
I have tested this code on my own laptop.

## Results

This shows a step of the example of a gaussian distributed water surface.
![Shallow water equations example](/docs/figs/part2/swe.png)

### The physics you are resolving

### Performance

#### Memory throughput

#### Weak scaling

#### Work-precision diagrams

## Discussion

## References

https://en.wikipedia.org/wiki/Shallow_water_equations#cite_note-3
https://users.oden.utexas.edu/~arbogast/cam397/dawson_v2.pdf
