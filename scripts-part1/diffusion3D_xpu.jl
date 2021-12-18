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
Main fucntion of diffusion solver.
"""
function pseudoStep!(C, D, dx, dy, dz,nx,ny,nz, dt)

    delta_tau  = min(dx,dy,dz)^2/D/8.1
    maxiter = 1e5
    @show delta_tau

    C_tau = copy(C)
    @show size(C_tau)
    dCdtTau = @zeros(nx-2,ny-2,nz-2)
    ResC = @zeros(nx-2,ny-2,nz-2)


    qx     = @zeros(nx-1,ny-2, nz-2)
    qy     = @zeros(nx-2,ny-1, nz-2)
    qz     = @zeros(nx-2,ny-2, nz-1)

    damp = 1.0 - 2π/nx
    iter = 0

    while iter < maxiter


        @parallel computeStep!(qx,qy,qz,C, ResC, dCdtTau, C_tau, dx, dy, dz, D,delta_tau,damp, dt)
        
        @show norm(ResC)/sqrt(nx*ny*nz)
        ((norm(ResC)/sqrt(nx*ny*nz)) > 1e-10) || break

        iter += 1
    end  

    C_tau
end

@views function diffusion3D(do_visu=true)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    D          = 1.0              # diffusion coefficient
    ttot       = 1.0             # total simulation time
    dt         = 0.2              # physical time step
    # Numerics
    nz = ny = nx = 32 
    nout   = 100
    # Derived numerics
    dx, dy, dz = lx/nx, ly/ny, lz/nz 
    #dt     = min(dx, dy, dz)^2/D/6.1
    nt     = cld(ttot, dt)
    xc = LinRange(dx/2, lx-dx/2, nx)
    yc = LinRange(dy/2, ly-dy/2, ny)
    zc = LinRange(dz/2, lz-dz/2, nz)

    # Array initialisation
    C = @zeros(nx,ny,nz)
    for x in 1:nx
        for y in 1:ny
            for z in 1:nz
                C[x,y,z] = 2 * exp(-(xc[x] - lx/2)^2 -(yc[y] - ly/2)^2 -(zc[z] - lz/2)^2)
            end
        end
    end

    # Time loop
    for it = 1:nt
        
        #@show it
        C .= pseudoStep!(C, D, dx, dy, dz,nx,ny,nz, dt)


        if 0 == 0
            @show it
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:davos,clims=(0.0, 1.0), xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
            display(heatmap(xc, yc, C[:,Int(ny/2),:]'; opts...))
            
        end
    end
    return C
end
#diffusion3D(do_visu=true)
