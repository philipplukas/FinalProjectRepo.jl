using Plots
using IJulia

"""
Main fucntion of diffusion solver.
"""
@views function diffusion3D(nx; do_visu = false)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    D          = 1.0              # diffusion coefficient
    ttot       = 100              # total simulation time
    dt         = 0.2              # physical time step
    # Numerics
    ny = nz = nx
    nout   = 100
    # Derived numerics
    dx, dy, dz = lx/nx, ly/ny, lz/nz 
    dt     = min(dx, dy, dz)^2/D/6.1
    nt     = cld(ttot, dt)
    xc = LinRange(dx/2, lx-dx/2, nx)
    yc = LinRange(dx/2, lx-dx/2, nx)
    zc = LinRange(dy/2, ly-dy/2, ny)

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

    dCdt   = zeros(Float64, nx-2,ny-2, nz-2)
    qx     = zeros(Float64, nx-1,ny-2, nz-2)
    qy     = zeros(Float64, nx-2,ny-1, nz-2)
    qz     = zeros(Float64, nx-2,ny-2, nz-1)
    # Time loop
    for it = 1:nt
        qx         .= .-D.*diff(C[:,2:end-1, 2:end-1],dims=1)./dx
        qy         .= .-D.*diff(C[2:end-1,:, 2:end-1],dims=2)./dy
        qz         .= .-D.*diff(C[2:end-1, 2:end-1, :],dims=3)./dz
        dCdt       .= .-(diff(qx,dims=1)./dx .+ diff(qy,dims=2)./dy .+ diff(qz,dims=3)./dz)
        C[2:end-1,2:end-1,2:end-1] .= C[2:end-1,2:end-1,2:end-1] .+ dt.*dCdt
        if do_visu && (it % nout == 0)
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:davos,clims=(0.0, 1.0), xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
            display(heatmap(xc, yc, C[:,Int(ny/2),:]'; opts...))
        end
    end
    return C
end