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

## Results
Results section

### 3D diffusion
![diffusion animation](/docs/figs/part1/diffusion_3D_mxpu.gif)

The animation above is generated using the grid size nx = ny = nx = 32.
The middle shows the x/y surface in the middle of the z-direction at x = 5m.
The top image shows the x/y plane at x = 2.5, and the bottom image at x = 7.5.
Since we have a gaussian sphere, the middle one is the brightest image while the brigth circlec in the top and bottom images are smaller.

### Performance

The memory throughput is used to evaluate the performance of the code. The CPU based platform seems to be computer bound for this particular physics problem.

#### Memory throughput

The below diagram shows the memory throughput on the CPU setup for grid sizes nx = ny = nz = 16,32,128,256,512.
We can see a significant drop in performance after grid side length 128. The reason for this could that the calculation of the error is very compte intensive. 
It is therefore likely that we are computer bound for the diffusion problem on the CPU based setup.
It is very likely that this changes once we move to a GPU based setup.

![Memory throughput CPU](/docs/figs/part1/mem_throughput_cpu.png)

#### Weak scaling
Multi-GPU weak scaling

#### Work-precision diagrams
Provide a figure depicting convergence upon grid refinement; report the evolution of a value from the quantity you are diffusing for a specific location in the domain as function of numerical grid resolution. Potentially compare against analytical solution.

Provide a figure reporting on the solution behaviour as function of the solver's tolerance. Report the relative error versus a well-converged problem for various tolerance-levels. 

## Discussion
Discuss and conclude on your results

## References
Provide here refs if needed.
