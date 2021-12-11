using Plots
using IJulia


@views function avX(A) 
    return ((A[1:end-1, :] .+ A[2:end, :]) ./ 2) 
end

@views function avY(A) 
    return ((A[:, 1:end-1] .+ A[:,2:end]) ./ 2) 
end

"""
Main fucntion of diffusion solver.
"""
@views function diffusion3D(nx; do_visu = false)
    # Physics
    lx, ly = 100.0, 100.0       # domain size
    ttot       = 100              # total simulation time
    
    rho = 997 # Density of water
    g = 9.81# gravitaitonal acceleration
    # Numerics
    ny = nz = nx
    nout   = 10
    # Derived numerics
    dx, dy = lx/nx, ly/ny
    
    xc = LinRange(dx/2, lx-dx/2, nx)
    yc = LinRange(dy/2, ly-dy/2, ny)

    # Array initialisation
    H =  20 .* exp.(.-((xc .- (lx/2))./3).^2 .-((yc' .- (ly/2))./3).^2 ) .+ 20

    # CFL condition accroding to https://aip.scitation.org/doi/abs/10.1063/1.4940835
    dt = 0.1 * min(dx,dy) / sqrt(maximum(H)*g)
    nt     = cld(ttot, dt)
    @show dt
    #return
 
    u = zeros(Float64, nx+1, ny)
    v = zeros(Float64, nx, ny+1)
    u_temp = zeros(Float64, nx+1, ny)
    v_temp = zeros(Float64, nx, ny+1)

    # Time loop
    for it = 1:nt

        #@show size(avY(H))
        #@show size(avX(avY(u)))
        #@show size(diff((rho .* avY(H) .* avX(avY(u)).^2),dims=1))
        #@show size(diff((rho .* avY(H) .* avX(avY(u)).^2) .+ (0.5 .* rho .* g .* avY(H).^2) , dims=1)./dx)
        #@show size(diff(rho .* avX(H) .* u[2:end-1,:] .* avX(avY(v)), dims=2)./dy)
        #@show size((
        #    diff((rho .* avY(H) .* avX(avY(u)).^2) .+ (0.5 .* rho .* g .* avY(H).^2) , dims=1)./dx .+
        #    diff(rho .* avX(H) .* u[2:end-1,:] .* avX(avY(v)), dims=2)./dy
        #))
        # Update of u and vector
        u_temp[2:end-1,2:end-1] .= u[2:end-1,2:end-1] .+ avY(
                    dt .* .-(
                        diff((rho .* avY(H) .* avX(avY(u)).^2) .+ (0.5 .* rho .* g .* avY(H).^2) , dims=1)./dx .+
                        diff(rho .* avX(H) .* u[2:end-1,:] .* avX(avY(v)), dims=2)./dy
                    ) ./ (rho .* avX(avY(H)))
                 )
        
        v_temp[2:end-1,2:end-1] .= v[2:end-1,2:end-1] .+ avX(
                    dt .* .-(
                    diff(rho .* avY(H) .* avX(avY(u)) .* v[:,2:end-1] , dims=1)./dx .+
                    diff((rho .* avX(H) .* avX(avY(v)).^2) .+ (0.5 .* rho .* g .* avX(H).^2) , dims=2)./dy
                    ) ./ (rho .*  avX(avY(H)))
                )


        u .= u_temp
        v .= v_temp


        #@show size(avX(H[:,2:end-1]))
        #@show size(u)
        #@show size(diff(rho .* avX(H[:,2:end-1]) .* u[2:end-1,2:end-1], dims=1))
        H[2:end-1,2:end-1] .= H[2:end-1,2:end-1] .+ dt .* .-(
                            diff(rho .* avX(H[:,2:end-1]) .* u[2:end-1,2:end-1], dims=1)./ dx .+
                            diff(rho .* avY(H[2:end-1,:]) .* v[2:end-1,2:end-1], dims=2)./ dy
                        ) ./ rho

        @show it
        #display(heatmap(u_temp'))
        #display(heatmap(v_temp'))
        if it % nout == 0
            #display(heatmap(H'))
            display(surface(xc,yc,H',zlims=(0,30)))
            #update!(scene)
            #sleep(0.3)
            
        end
        
        #@show u_temp
        #@show v_temp
        #dt = 0.5 * min(dx,dy) / sqrt(maximum(H)*g)
    end

end

diffusion3D(256)