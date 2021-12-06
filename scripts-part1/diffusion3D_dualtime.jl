using Plots
using IJulia

# Part 1 of final project: Diffusion equation
@views function init()
    nx_g = 30             # number of global grid points
    Xc_g = zeros(nx_g, 1) # global coord vector
    H_g  = zeros(nx_g, 1) # global solution as obtained by implicitGlobalGrid's `gather!()`
    inds = Int.(ceil.(LinRange(1, length(Xc_g), 12)))
    Xc_g[inds] .= [0.46875, 1.40625, 2.34375, 2.96875, 3.90625, 4.84375, 5.46875, 6.40625, 7.34375, 7.96875, 8.90625, 9.53125]
    H_g[inds]  .= [1.2981288953788742e-6, 6.258528479372478e-6, 1.4557797727681202e-5, 2.2141755782992116e-5, 3.3990701348812344e-5, 4.0487959896720236e-5, 3.931867494216368e-5, 3.028472793911268e-5, 1.819088664977141e-5, 1.1341281028367691e-5, 4.326911919257118e-6, 1.2981288953788742e-6]
    return Xc_g, H_g
end

"""
Compute step of iterative solver.
"""
@views function compute_step()
    return
end

"""
Main fucntion of diffusion solver.
"""
function pseudoStep!(C, D, dx, dy, dz,nx,ny,nz, dt)

    delta_tau = 0.001

    C_tau = copy(C)
    dCdtTau = zeros(Float64, nx-2,ny-2,nz-2)


    qx     = zeros(Float64, nx-1,ny-2, nz-2)
    qy     = zeros(Float64, nx-2,ny-1, nz-2)
    qz     = zeros(Float64, nx-2,ny-2, nz-1)

    while true 

        qx         .= .-D.*diff(C_tau[:,2:end-1, 2:end-1],dims=1)./dx
        qy         .= .-D.*diff(C_tau[2:end-1,:, 2:end-1],dims=2)./dy
        qz         .= .-D.*diff(C_tau[2:end-1, 2:end-1, :],dims=3)./dz

        dCdtTau       .= ((C_tau[2:end-1,2:end-1,2:end-1] - C[2:end-1,2:end-1,2:end-1])/dt) .- (diff(qx,dims=1)./dx .+ diff(qy,dims=2)./dy .+ diff(qz,dims=3)./dz)
        C_tau[2:end-1,2:end-1,2:end-1] .= C_tau[2:end-1,2:end-1,2:end-1] .+ delta_tau.*dCdtTau
       # @show C_tau_next

        @show maximum(abs.(C_tau))
        (maximum(abs.(C_tau)) > 0.001) || break

    end  

    C = C_tau_next
    return C 
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
                C[x,y,z] = exp(-(xc[x] - lx/2)^2 -(yc[y] - ly/2)^2 -(zc[z] - lz/2)^2)
            end
        end
    end

    dCdt   = zeros(Float64, nx-2,ny-2, nz-2)
    qx     = zeros(Float64, nx-1,ny-2, nz-2)
    qy     = zeros(Float64, nx-2,ny-1, nz-2)
    qz     = zeros(Float64, nx-2,ny-2, nz-1)
    # Time loop
    for it = 1:nt
        
        #@show it
        C = pseudoStep!(C, D, dx, dy, dz,nx,ny,nz, dt)

        if 0 == 0
            @show it
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:davos,clims=(0.0, 1.0), xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
            display(heatmap(xc, yc, C[:,Int(ny/2),:]'; opts...))
            
        end
    end
    return
end
diffusion3D(do_visu=true)
