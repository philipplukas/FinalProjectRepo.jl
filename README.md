# Final Project Repository

[![Build Status](https://github.com/philipplukas/FinalProjectRepo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/philipplukas/FinalProjectRepo.jl/actions/workflows/CI.yml?query=branch%3Amain)

This semester, I have taken the course #Solving Partial Differential Equations in parallel on GPUs [Course website](https://eth-vaw-glaciology.github.io/course-101-0250-00/) for the fall semester 2021. The course is taught by Ludovic Räss, Mauro Werder & Samuel Omlin by the laboratory of Hydraulics, Hydrology, and Glaciology. In the course, we learn about physical Ordinary Differential Equations, about the programming Julia, and how to solve ODE's numerical thrugh Finite Differences on Multi XPU's. We learnt aout paralle computing based on GPU's and MPI and general optimization technique and performance measurements. 

In the least part of the course, we had to work on a final porject which consits of two parts.
In [**Part-1**](/docs/part1.md) I implemented a multi-XPU solver for diffusion in 3D. 
For [**Part-2**](/docs/part2.md), each student has to choose a PDE individually to work on. I choose to implement a solve for the Shallow Water Equations in two dimensions. Further, I studied the example application to the dam-break problem.


