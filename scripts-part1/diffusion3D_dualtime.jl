using Plots
using IJulia
using LinearAlgebra

"""
Main fucntion of diffusion solver.
"""
function pseudoStep!(C, D, dx, dy, dz,nx,ny,nz, dt)

    delta_tau  = min(dx,dy,dz)^2/D/6.1
    #delta_tau = (1.0/(min(dx,dy,dz)^2/D/6.1) + 1.0/dt)^-1
    
    maxiter = 1e5
    @show delta_tau

    C_tau = copy(C)
    dCdtTau = zeros(Float64, nx-2,ny-2,nz-2)
    ResC = zeros(Float64, nx-2,ny-2,nz-2)


    qx     = zeros(Float64, nx-1,ny-2, nz-2)
    qy     = zeros(Float64, nx-2,ny-1, nz-2)
    qz     = zeros(Float64, nx-2,ny-2, nz-1)

    damp = 1.0 - 29/nx
    iter = 0

    while iter < maxiter

        qx         .= .-D.*diff(C_tau[:,2:end-1, 2:end-1],dims=1)./dx
        qy         .= .-D.*diff(C_tau[2:end-1,:, 2:end-1],dims=2)./dy
        qz         .= .-D.*diff(C_tau[2:end-1, 2:end-1, :],dims=3)./dz
        ResC        .= .-((C_tau[2:end-1,2:end-1,2:end-1] .- C[2:end-1,2:end-1,2:end-1])./dt) .- (diff(qx,dims=1)./dx .+ diff(qy,dims=2)./dy .+ diff(qz,dims=3)./dz)
        dCdtTau       .= ResC .+ damp .* dCdtTau
        C_tau[2:end-1,2:end-1,2:end-1] .= C_tau[2:end-1,2:end-1,2:end-1] .+ delta_tau.*dCdtTau

        @show norm(ResC)/sqrt(nx*ny*nz)
        ((norm(ResC)/sqrt(nx*ny*nz)) > 1e-8) || break

        iter += 1
    end  

    return C_tau
end

@views function diffusion3D(do_visu=false)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    D          = 1.0              # diffusion coefficient
    ttot       = 1.0              # total simulation time
    dt         = 0.2              # physical time step
    # Numerics
    ny = nz = nx = 32
    nout   = 100
    # Derived numerics
    dx, dy, dz = lx/nx, ly/ny, lz/nz 
    #dt     = min(dx, dy, dz)^2/D/6.1
    nt     = cld(ttot, dt)
    xc = LinRange(dx/2, lx-dx/2, nx)
    yc = LinRange(dy/2, ly-dy/2, ny)
    zc = LinRange(dz/2, lz-dz/2, nz)

    # Array initialisation
    C = zeros(nx,ny,nz)
    for x in 1:nx
        for y in 1:ny
            for z in 1:nz
                C[x,y,z] = 2 * exp(-(xc[x] - lx/2)^2 -(yc[y] - ly/2)^2 -(zc[z] - lz/2)^2)
            end
        end
    end

    # Boundary conditions
    C[1,:,:] .= 0
    C[nx,:,:] .= 0
    C[:,1,:] .= 0
    C[:,ny,:] .= 0
    C[:,:,1] .= 0
    C[:,:,nz] .= 0

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

diffusion3D()