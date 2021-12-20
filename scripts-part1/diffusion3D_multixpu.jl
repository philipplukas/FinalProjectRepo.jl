# Run as MPI-program:
# ~/.julia/bin/mpiexecjl -n 1 julia --project diffusion3D_multixpu.jl


using Plots
using Printf
using IJulia
using LinearAlgebra

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using ImplicitGlobalGrid
using MPI


# MPI based norm function too calculate error on the global grid.
norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))


#Reference
#https://github.com/eth-cscs/ImplicitGlobalGrid.jl

"""
Main compute step function using ParallelStencil's @parallels.
Uses damping to speed up converging process.
Meaning of parameters can be looked up in the diffusion3D pseudoStep! funciton where they are defiend.
The names are kept the same.
"""
@parallel function computeStep!(qx,qy,qz,C, ResC, dCdtTau, C_tau, dx, dy, dz,D, delta_tau, damp, dt)
    @all(qx)   = -D*@d_xi(C_tau)/dx
    @all(qy)   = -D*@d_yi(C_tau)/dy
    @all(qz)   = -D*@d_zi(C_tau)/dz

    @all(ResC) = -((@inn(C_tau) - @inn(C))/dt) - (@d_xa(qx)/dx + @d_ya(qy)/dy + @d_za(qz)/dz)
    @all(dCdtTau)    = @all(ResC) + damp * @all(dCdtTau)
    @inn(C_tau) = @inn(C_tau) + delta_tau.*@all(dCdtTau)
    return
end

"""
One pseudo step corresponds to one time step.
Each pseudo step ams for a steady state solution for a fixed time according to time C was produces.
"""
@views function pseudoStep!(C, D, dx, dy, dz,nx,ny,nz, dt)

    # This parameter alters the speed of the algoirthm.
    # For physical correctnes, it needs to be set according to the CFL condition.
    delta_tau  = min(dx,dy,dz)^2/D/8.1

    # In case the error doesn't converge towards zero.
    maxiter = 1e5
    @show delta_tau

    # C is kept fixed and corresponds to time t0
    # C_tau is iterated on and will hold the solution for time t0 + dt upon convergence.
    C_tau = copy(C)

    # Variables storing intermediate results of each iteration.
    dCdtTau = @zeros( nx-2,ny-2,nz-2)
    ResC = @zeros( nx-2,ny-2,nz-2)

    # Fluxes in all direction of the measured quantity (Pressure/concentration, etc...)
    qx     = @zeros( nx-1,ny-2, nz-2)
    qy     = @zeros( nx-2,ny-1, nz-2)
    qz     = @zeros( nx-2,ny-2, nz-1)


    # Choosing ad damping parameter which changes with the grid size.
    damp = 1.0 - 29/nx




    # How often error will be measured; Shouldn't be calculated every iteration for performance reasons.
    nout = 10

    iter = 0
    err = Inf

    while (iter < maxiter) && (err > 1e-13)

        @parallel computeStep!(qx,qy,qz,C, ResC, dCdtTau, C_tau, dx, dy, dz, D,delta_tau,damp, dt)
        
        #@show norm(ResC)/sqrt(nx*ny*nz)

        if iter % nout == 0
            err = norm_g(ResC)/sqrt(nx_g()*ny_g()*nz_g())
            #@show err
        end
        
        iter += 1
    end  

    return C_tau
end

@views function diffusion3D(nx;do_visu=true)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    D          = 1.0              # diffusion coefficient
    ttot       = 1.0              # total simulation time
    dt         = 0.2              # physical time step


    # Numerics
    ny = nz = nx

    # Values will be plotted every nout' iteration.
    nout   = 1


    me, dims = init_global_grid(nx, ny, nz)  # Initialization of MPI and more...
    @static if USE_GPU select_device() end  # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy, dz  = lx/nx_g(), ly/ny_g(), lz/nz_g()
    

    nt     = cld(ttot, dt)

    # Initalize gaussian with amplitude 2 and standard deviation of 1.
    C = @zeros(nx,ny,nz)
    C .= Data.Array([2 * exp(-(x_g(ix,dx,C) - lx/2)^2 -(y_g(iy,dy,C) - ly/2)^2 -(z_g(iz,dz,C) - lz/2)^2) for ix in 1:nx, iy in 1:ny, iz in 1:nz ])

    # Boundary conditions

    # Check if y/z plane is touching the boundary 
    #if 1 == x_g(2, dx, C)
    #    C[2,:,:] .= 0
    #end
    #if nx_g() == x_g(nx-1, dx, C)
    #    C[nx-1,:,:] .= 0
    #end

    # Check if x/z plane is touching the boundary 
    #if 1 == y_g(2, dy, C)
    #    C[:,2,:] .= 0
    #end
    #if ny_g() == y_g(ny-1, dy, C)
    #    C[:,ny-1,:] .= 0
    #end

    # Check if x/y plane is touching the boundary 
    #if 1 == z_g(2, nz, C)
    #    C[:,:,2] .= 0
    #end
    #if nz_g() == z_g(nz-1, dz, C)
    #    C[:,:,nz-2] .= 0
    #end

    # Adapted from:
    # https://eth-vaw-glaciology.github.io/course-101-0250-00/lecture8/
    # Setup for animation 
    if do_visu
        if (me==0) ENV["GKSwstype"]="nul"; if isdir("viz3D_mxpu_out")==false mkdir("viz3D_mxpu_out") end; loadpath = "./viz3D_mxpu_out/"; anim = Animation(loadpath,String[]); println("Animation directory: $(anim.dir)") end
        nx_v, ny_v, nz_v  = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]
        if (nx_v*ny_v*nz_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
        C_v   = zeros(nx_v, ny_v, nz_v) # global array for visu
        C_inn = zeros(nx-2, ny-2, nz-2) # no halo local array for visu
        Xi_g, Yi_g, Zi_g = LinRange(dx+dx/2, lx-dx-dx/2, nx_v), LinRange(dy+dy/2, ly-dy-dy/2, ny_v), LinRange(dz+dz/2, lz-dz-dz/2, nz_v) # inner points only
    end


    # Time loop
    for it = 1:nt

        C .= pseudoStep!(C, D, dx, dy, dz,nx,ny,nz, dt)

        # Visualize
        if do_visu && (it % nout == 0)
            C_inn .= C[2:end-1,2:end-1, 2:end-1]; gather!(C_inn, C_v)
            if (me==0)
                #display(scatter(Array(C_v), markersize=100, show_axis=true))
                opts = (aspect_ratio=1, xlims=(Xi_g[1], Xi_g[end]), ylims=(Yi_g[1], Yi_g[end]), clims=(0.0, 1.0), c=:davos, xlabel="Lx", ylabel="Ly")                #p = plot()
                #heatmap!(p, Hm..., layout=(3, 1);opts...)
                l = (3, 1)
                #title = plot(title = "time = $(round(it*dt, sigdigits=3))", grid = false, showaxis = false, bottom_margin = -50Plots.px)
                p1 = heatmap(Xi_g, Yi_g, Array(C_v[:,:,Int(round(0.4*nz_g()))])'; label="time = $(0.25*nz_g())", opts...)
                p2 = heatmap(Xi_g, Yi_g, Array(C_v[:,:,Int(round(0.5*nz_g()))])'; label="time = $(Int(0.5*nz_g()))", opts...)
                p3 = heatmap(Xi_g, Yi_g, Array(C_v[:,:,Int(round(0.6*nz_g()))])'; label="time = $(Int(0.75*nz_g()))", opts...)
                plot(p1, p2, p3, layout = l, title = "time = $(round(it*dt, sigdigits=3))")
                frame(anim)
            end
        end

        update_halo!(C);    

    end

    if (do_visu && me==0) gif(anim, "diffusion_3D_mxpu.gif", fps = 5)  end

    finalize_global_grid();

    #@show size(C_v)
end
diffusion3D(32, do_visu=true)
