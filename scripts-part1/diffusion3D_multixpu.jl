using Plots
using Printf
using IJulia
using LinearAlgebra
using CUDA
using Threads

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using ImplicitGlobalGrid


#Reference
#https://github.com/eth-cscs/ImplicitGlobalGrid.jl

@parallel function computeStep!(qx,qy,qz,C, ResC, dCdtTau, C_tau, dx, dy, dz,D, delta_tau, damp, dt)
    @all(qx)   = -D*@d_xa(C_tau)/dx
    @all(qy)   = -D*@d_ya(C_tau)/dy
    @all(qz)   = -D*@d_za(C_tau)/dz
    @all(ResC) = -((@inn(C_tau) - @inn(C))/dt) - (@d_xi(qx)/dx + @d_yi(qy)/dy + @d_zi(qz)/dz)
    @all(dCdtTau)    = @all(ResC) + damp * @all(dCdtTau)
    @inn(C_tau) = @inn(C_tau) + delta_tau.*@all(dCdtTau)
    return
end

"""
Main fucntion of diffusion solver.
"""
function pseudoStep!(C, D, dx, dy, dz,nx,ny,nz, dt)

    delta_tau  = min(dx^2,dy^2,dz^2)/D/8.1
    maxiter = 1e5
    @show delta_tau

    C_tau = copy(C)
    dCdtTau = zeros(Float64, nx-2,ny-2,nz-2)
    ResC = zeros(Float64, nx-2,ny-2,nz-2)


    qx     = zeros(Float64, nx-1,ny-2, nz-2)
    qy     = zeros(Float64, nx-2,ny-1, nz-2)
    qz     = zeros(Float64, nx-2,ny-2, nz-1)

    damp = 1.0 - 2π/nx
    iter = 0

    while iter < maxiter


        computeStep!(qx,qy,qz,C, ResC, dCdtTau, C_tau, dx, dy, dz, D,delta_tau,damp, dt)
        
        @show norm(ResC)/sqrt(nx*ny*nz)
        ((norm(ResC)/sqrt(nx*ny*nz)) > 1e-8) || break

        iter += 1
    end  

    C_tau
end

@views function diffusion3D(;do_visu=true)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    D          = 1.0              # diffusion coefficient
    ttot       = 100              # total simulation time
    dt         = 0.2              # physical time step
    # Numerics
    nx, ny, nz = 128, 128, 128
    nout   = 100


    me, dims = init_global_grid(nx, ny, nz)  # Initialization of MPI and more...
    @static if USE_GPU select_device() end  # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy, dz  = Lx/nx_g(), Ly/ny_g(), Lz/nz_g()

    # Derived numerics
    #dx, dy, dz = lx/nx, ly/ny, lz/nz 
    #dt     = min(dx, dy, dz)^2/D/6.1
    nt     = cld(ttot, dt)
    xc = LinRange(dx/2, lx-dx/2, nx)
    yc = LinRange(dx/2, lx-dx/2, nx)
    zc = LinRange(dy/2, ly-dy/2, ny)

    # Array initialisation
    C = CUDA.zeros(nx,ny,nz)
    for x in 1:nx
        for y in 1:ny
            for z in 1:nz
                C[x,y,z] = 2 * @exp(-(x_g(ix,dx,C) - lx/2)^2 -(y_g(iy,dy,C) - ly/2)^2 -(z_g(iz,dz,C) - lz/2)^2)
            end
        end
    end

    dCdt   = CUDA.zeros(Float64, nx-2,ny-2, nz-2)
    qx     = CUDA.zeros(Float64, nx-1,ny-2, nz-2)
    qy     = CUDA.zeros(Float64, nx-2,ny-1, nz-2)
    qz     = CUDA.zeros(Float64, nx-2,ny-2, nz-1)
    # Time loop
    for it = 1:nt
        
        @show it
        C .= pseudoStep!(C, D, dx, dy, dz,nx,ny,nz, dt)


        #if 0 == 0
        #    @show it
        #    opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:davos,clims=(0.0, 1.0), xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
        #    display(heatmap(xc, yc, C[:,Int(ny/2),:]'; opts...))
        #    
        #end

        update_halo!(C);    
    end

    finalize_global_grid();
    return
end
diffusion3D(do_visu=true)
