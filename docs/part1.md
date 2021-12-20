# Part 1: 3D multi-XPUs diffusion solver

## Intro
In this part of the project a solver for 3D diffusion implemented. 
The forward euler method using pseudo-trancient acceleratin is used. 
This dual-time approach incoroporates both, phsycisal and pseudo time.
Approching the diffusion equations allows for faster converage,
but also uncoupling of the physical time from the actual numerical timestepping.

<img width="200" alt="Screenshot 2021-12-20 at 10 37 47" src="https://user-images.githubusercontent.com/18243049/146793201-b71442c3-dd88-4467-96ec-88560559ee4e.png">

This eqution describes the diffusion in multiple dimensions; D is the diffusion coefficient and H describes the property which is being diffused. THis could be pressure or concentration. Since we are tackling 3D diffusion, H has 3 dimensions in our soluion.
My aim is to implement a diffusion code for 3D dimensions and run the simulation using the following initial parameters:

```julia
lx, ly, lz = 10.0, 10.0, 10.0 # domain size
D          = 1.0              # diffusion coefficient
ttot       = 1.0              # total simulation time
dt         = 0.2              # physical time step
```


## Methods

To accomplish the (mulit-)XPU implementation, I rely on both libraryies, [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) and [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl). I use the `@parallel` type of kernel defintion since its implementation is the most straightforwards. 

As for the hardware, I am able to run the code on two steups, once on my own computer and once on the supercomputer of the University of Lucern.
My own laptop is CPU based and powered by a 2.3 GHz 8-Core Intel Core i9 processor. On the supercomputer, my code runs on a GPU based environment with the GPU Nvidia Tesla V100 32GB SXM2.




The methods to be used:
- spatial and temporal discretisation
- solution approach
- hardware
- ...

## Results
Results section

### 3D diffusion
Report an animation of the 3D solution here and provide and concise description of the results. _Unleash your creativity to enhance the visual output._

### Performance
Briefly elaborate on performance measurement and assess whether you are compute or memory bound for the given physics on the targeted hardware.

#### Memory throughput
Strong-scaling on CPU and GPU -> optimal "local" problem sizes.

#### Weak scaling
Multi-GPU weak scaling

#### Work-precision diagrams
Provide a figure depicting convergence upon grid refinement; report the evolution of a value from the quantity you are diffusing for a specific location in the domain as function of numerical grid resolution. Potentially compare against analytical solution.

Provide a figure reporting on the solution behaviour as function of the solver's tolerance. Report the relative error versus a well-converged problem for various tolerance-levels. 

## Discussion
Discuss and conclude on your results

## References
Provide here refs if needed.
