using Plots
using IJulia


@views function avX(A) 
    return ((A[1:end-1, :] .+ A[2:end, :]) ./ 2) 
end

@views function avY(A) 
    return ((A[:, 1:end-1] .+ A[:,2:end]) ./ 2) 
end

@views function avNbX(A) 
    return ((A[1:end-2, :] .+ A[3:end, :]) ./ 2) 
end

@views function avNbY(A) 
    return ((A[:, 1:end-2] .+ A[:,3:end]) ./ 2) 
end

"""
Main fucntion of diffusion solver.
"""
@views function diffusion3D(nx; do_visu = false)
    # Physics
    lx, ly = 100.0, 100.0       # domain size
    ttot       = 20            # total simulation time
    
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
    H = fill(Float64(20), nx, ny)
    H[1:div(nx,4),:] .= 20 + 20
    for i in 1:20
        H[div(nx,4)+i,:] .= 20 - i + 20
    end
    H[div(nx,4)+21:100,:] .= 0 + 20
    display(surface(xc[3:end-2,3:end-2],yc,H[3:end-2,3:end-2]',zlims=(0,80)))
    sleep(0.1)
    # CFL condition accroding to https://aip.scitation.org/doi/abs/10.1063/1.4940835
    dt = 0.05 * min(dx,dy) / sqrt(maximum(H)*g)
    @show dt
    nt     = cld(ttot, dt)
    @show dt
    #return
 
    u = zeros(Float64, nx+1, ny)
    v = zeros(Float64, nx, ny+1)
    u_temp = zeros(Float64, nx+1, ny)
    v_temp = zeros(Float64, nx, ny+1)

    # Time loop
    @show nt
    for it = 1:4000

        #@show size(avY(H))
        #@show size(avX(avY(u)))
        #@show size(diff((rho .* avY(H) .* avX(avY(u)).^2),dims=1))
        #@show size(diff((rho .* avY(H) .* avX(avY(u)).^2) .+ (0.5 .* rho .* g .* avY(H).^2) , dims=1)./dx)
        #@show size(diff(rho .* avX(H) .* u[2:end-1,:] .* avX(avY(v)), dims=2)./dy)
        #@show size((
        #    diff((rho .* avY(H) .* avX(avY(u)).^2) .+ (0.5 .* rho .* g .* avY(H).^2) , dims=1)./dx .+
        #    diff(rho .* avX(H) .* u[2:end-1,:] .* avX(avY(v)), dims=2)./dy
        #))
        

        u_temp[3:end-2,3:end-2] .= avNbX(u[2:end-1,3:end-2]) .+ (
                    dt .* .-(
                        avNbX(diff((rho .* avY(H) .* avX(avY(u)).^2) .+ (0.5 .* rho .* g .* avY(H).^2) , dims=1))[:,2:end-1]./dx .+
                        avNbX(diff(rho .* avX(H) .* u[2:end-1,:] .* avX(avY(v)), dims=2))[:,2:end-1]./dy
                    ) ./ (rho .* avX(avY(H)))[2:end-1,2:end-2]

                    .+

                    dt .* .-(
                        avNbX(diff((rho .* avY(H) .* avX(avY(u)).^2) .+ (0.5 .* rho .* g .* avY(H).^2) , dims=1))[:,2:end-1]./dx .+
                        avNbX(diff(rho .* avX(H) .* u[2:end-1,:] .* avX(avY(v)), dims=2))[:,2:end-1]./dy
                    ) ./ (rho .* avX(avY(H)))[2:end-1,3:end-1]
                 ) ./2
         
        #v_temp[3:end-2,3:end-2] .= avNbY(v[3:end-2,2:end-1]) .+ avX(
        #            dt .* .-(
        #            avNbY(diff(rho .* avY(H) .* avX(avY(u)) .* v[:,2:end-1] , dims=1))[2:end-1,:]./dx .+
        #            avNbY(diff((rho .* avX(H) .* avX(avY(v)).^2) .+ (0.5 .* rho .* g .* avX(H).^2) , dims=2))[2:end-1,:]./dy
        #            ) ./ (rho .*  avX(avY(H)))[2:end-1,2:end-1]
        #        )


        u .= u_temp
        #v .= v_temp


        #@show size(avX(H[:,2:end-1]))
        #@show size(u)
        #@show size(diff(rho .* avX(H[:,2:end-1]) .* u[2:end-1,2:end-1], dims=1))
        @show size(diff(rho .* avX(H[:,2:end-1]) .* u[2:end-1,2:end-1], dims=1))
        H[3:end-2,3:end-2] .= H[3:end-2,3:end-2] .+ dt .* .-(
                            avNbX(diff(rho .* avX(H[:,2:end-1]) .* u[2:end-1,2:end-1], dims=1))[:,2:end-1]./ dx #.+
                            #avNbY(diff(rho .* avY(H[2:end-1,:]) .* v[2:end-1,2:end-1], dims=2))[2:end-1,:]./ dy
                        ) ./ rho
        H[1,:] .= H[3,:]
        H[2,:] .= H[3,:]
        H[end-1,:] .= H[end-2,:]
        H[end,:] .= H[end-2,:]

        H[:,1] .= H[:, 3]
        H[:,2] .= H[:, 3]
        H[:,end-1] .= H[:, end-2]
        H[:,end] .= H[:, end-2]

        #H[1:2,1:2] .= H[3,3]
        #H[1:2,end-1:end] .= H[3,end-2]
        #H[end-1:end,1:2] .= H[end-2,3]
        #H[end-1:end, end-1:end] .= H[end-2,end-2]


        @show it
        #display(heatmap(u_temp'))
        #display(heatmap(v_temp'))
        if it % nout == 0
            #display(heatmap(H'))
            display(surface(xc[3:end-2],yc[3:end-2],H[3:end-2,3:end-2]',zlims=(0,80)))
            #update!(scene)
            #sleep(0.3)
            
        end
        
        #@show u_temp
        #@show v_temp
        #dt = 0.5 * min(dx,dy) / sqrt(maximum(H)*g)
    end

end

diffusion3D(256)