using Plots
using IJulia

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

#https://en.wikipedia.org/wiki/Lax%E2%80%93Friedrichs_method

@parallel_indices (ix, iy) function compute_step!(H, u, v, dtu_x, dtu_y, dtv_x, dtv_y, nx, ny, g, rho, dt, dx, dy)
    #for ix in 1:nx
    #    for iy in 1:ny
            if (ix < nx-2) && (iy < ny-2)
            dtu_x[ix, iy] = ((
                (((rho * ((H[ix+1,iy+1]+H[ix+1,iy+2])/2) * ((u[ix+1,iy+1] + u[ix+2,iy+1] + u[ix+1, iy+2] + u[ix+2,iy+2])/4)^2) + (0.5 * rho * g * ((H[ix+1,iy+1]+H[ix+1,iy+2])/2)^2)) -
                ((rho * ((H[ix,iy+1]+H[ix,iy+2])/2) * ((u[ix,iy+1] + u[ix+1,iy+1] + u[ix, iy+2] + u[ix+1,iy+2])/4)^2) + (0.5 * rho * g * ((H[ix,iy+1]+H[ix,iy+2])/2)^2)))
                +
                (((rho * ((H[ix+3,iy+1]+H[ix+3,iy+2])/2) * ((u[ix+3,iy+1] + u[ix+4,iy+1] + u[ix+3, iy+2] + u[ix+4,iy+2])/4)^2) + (0.5 * rho * g * ((H[ix+3,iy+1]+H[ix+3,iy+2])/2)^2)) -
                ((rho * ((H[ix+2,iy+1]+H[ix+2,iy+2])/2) * ((u[ix+2,iy+1] + u[ix+3,iy+1] + u[ix+2, iy+2] + u[ix+3,iy+2])/4)^2) + (0.5 * rho * g * ((H[ix+2,iy+1]+H[ix+2,iy+2])/2)^2)))
            ) / 2
            ) /dx
        
    #dtu_y = avNbX(diff(rho .* avX(H) .* u[2:end-1,:] .* avX(avY(v)), dims=2))[:,2:end-1]./dy
    #dtu_y .= ((
    #    ((rho .* ((H[1:end-3,3:end-1] .+ H[2:end-2,3:end-1]) ./ 2) .* u[2:end-3,3:end-1] .* ((v[1:end-3,3:end-2] .+ v[2:end-2,3:end-2] .+ v[1:end-3,4:end-1] .+ v[2:end-2,4:end-1]) ./ 4)) .-
    #    (rho .* ((H[1:end-3,2:end-2] .+ H[2:end-2,2:end-2]) ./ 2) .* u[2:end-3,2:end-2] .* ((v[1:end-3,2:end-3] .+ v[2:end-2,2:end-3] .+ v[1:end-3,3:end-2] .+ v[2:end-2,3:end-2]) ./ 4)))
    #    .+
    #    ((rho .* ((H[3:end-1,3:end-1] .+ H[4:end,3:end-1]) ./ 2) .* u[4:end-1,3:end-1] .* ((v[3:end-1,3:end-2] .+ v[4:end,3:end-2] .+ v[3:end-1,4:end-1] .+ v[4:end,4:end-1]) ./ 4)) .-
    #    (rho .* ((H[3:end-1,2:end-2] .+ H[4:end,2:end-2]) ./ 2) .* u[4:end-1,2:end-2] .* ((v[3:end-1,2:end-3] .+ v[4:end,2:end-3] .+ v[3:end-1,3:end-2] .+ v[4:end,3:end-2]) ./ 4)))
    #) ./ 2
    #) ./dy

            dtu_y[ix, iy] = ((
                ((rho * ((H[ix,iy+2] + H[ix+1,iy+2]) / 2) * u[ix+1,iy+2] * ((v[ix,iy+2] + v[ix+1,iy+2] + v[ix,iy+3] + v[ix+1,iy+3]) / 4)) -
                (rho * ((H[ix,iy+1] + H[ix+1,iy+1]) / 2) * u[ix+1,iy+1] * ((v[ix,iy+1] + v[ix+1,iy+1] + v[ix,iy+2] + v[ix+1,iy+2]) / 4)))
                +
                ((rho * ((H[ix+2,iy+2] + H[ix+3,iy+2]) / 2) * u[ix+3,iy+2] * ((v[ix+2,iy+2] + v[ix+3,iy+2] + v[ix+2,iy+3] + v[ix+3,iy+3]) / 4)) -
                (rho * ((H[ix+2,iy+1] + H[ix+3,iy+1]) / 2) * u[ix+3,iy+1] * ((v[ix+2,iy+1] + v[ix+3,iy+1] + v[ix+2,iy+2] + v[ix+3,iy+2]) / 4)))
            ) / 2
            ) /dy
            end


    #@show size(dtu_y)
    #return

            if (ix > 2) && (ix < nx-1) && (iy > 2) && (iy < ny-1)
            u[ix,iy] = (u[ix-1,iy] + u[ix+1,iy])/2 + (
                        (dt * -(
                            dtu_x[ix-1,iy-2] +
                            dtu_y[ix-1,iy-2]
                        ) / (rho * (H[ix,iy-1] + H[ix,iy] + H[ix+1,iy-1] + H[ix+1,iy])/4))

                        + 

                        (dt * -(
                            dtu_x[ix-2,iy-2] +
                            dtu_y[ix-2,iy-2]
                        ) / (rho * (H[ix-1,iy-1] + H[ix-1,iy] + H[ix,iy-1] + H[ix,iy])/4))
                    ) /2
            end


            if (ix < nx-2) && (iy < ny-2)
    #dtv_x .= ((
    #    ((rho .* ((H[3:end-1,1:end-3] .+ H[3:end-1,2:end-2]) ./ 2) .* ((u[3:end-2,1:end-3] .+ u[3:end-2,2:end-2] .+ u[4:end-1, 1:end-3] .+ u[4:end-1, 2:end-2]) ./4) .* v[3:end-1,2:end-3]) .-
    #    (rho .* ((H[2:end-2,1:end-3] .+ H[2:end-2,2:end-2]) ./ 2) .* ((u[2:end-3,1:end-3] .+ u[2:end-3,2:end-2] .+ u[3:end-2, 1:end-3] .+ u[3:end-2, 2:end-2]) ./4) .* v[2:end-2,2:end-3]))
    #    .+ 
    #    ((rho .* ((H[3:end-1,3:end-1] .+ H[3:end-1,4:end]) ./ 2) .* ((u[3:end-2,3:end-1] .+ u[3:end-2,4:end] .+ u[4:end-1, 3:end-1] .+ u[4:end-1, 4:end]) ./4) .* v[3:end-1,4:end-1]) .-
    #    (rho .* ((H[2:end-2,3:end-1] .+ H[2:end-2,4:end]) ./ 2) .* ((u[2:end-3,3:end-1] .+ u[2:end-3,4:end] .+ u[3:end-2, 3:end-1] .+ u[3:end-2, 4:end]) ./4) .* v[2:end-2,4:end-1]))
    #) ./ 2
    #)./dx
    dtv_x[ix,iy] = ((
        ((rho * ((H[ix+2,iy] + H[ix+2,iy+1]) / 2) * ((u[ix+2,iy] + u[ix+2,iy+1] + u[ix+3, iy] + u[ix+3, iy+1]) /4) * v[ix+2,iy+1]) -
        (rho * ((H[ix+1,iy] + H[ix+1,iy+1]) / 2) * ((u[ix+1,iy] + u[ix+1,iy+1] + u[ix+2, iy] + u[ix+2, iy+1]) /4) * v[ix+1,iy+1]))
        + 
        ((rho * ((H[ix+2,iy+2] + H[ix+2,iy+3]) / 2) * ((u[ix+2,iy+2] + u[ix+2,iy+3] + u[ix+3, iy+2] + u[ix+3, iy+3]) /4) * v[ix+2,iy+3]) -
        (rho * ((H[ix+1,iy+2] + H[ix+1,iy+3]) / 2) * ((u[ix+1,iy+2] + u[ix+1,iy+3] + u[ix+2, iy+2] + u[ix+2, iy+3]) /4) * v[ix+1,iy+3]))
    ) / 2
    )/dx
            end


    #dtv_y = ((
    #    (((rho .* ((H[2:end-2,2:end-2] .+ H[3:end-1,2:end-2]) ./ 2) .* ((v[2:end-2,2:end-3] .+ v[2:end-2,3:end-2] .+ v[3:end-1,2:end-3] .+ v[3:end-1,3:end-2]) ./ 4).^2) .+ (0.5 .* rho .* g .* ((H[2:end-2,2:end-2] .+ H[3:end-1,2:end-2]) ./ 2).^2)) .-
    #    ((rho .* ((H[2:end-2,1:end-3] .+ H[3:end-1,1:end-3]) ./ 2) .* ((v[2:end-2,1:end-4] .+ v[2:end-2,2:end-3] .+ v[3:end-1,1:end-4] .+ v[3:end-1,2:end-3]) ./ 4).^2) .+ (0.5 .* rho .* g .* ((H[2:end-2,1:end-3] .+ H[3:end-1,1:end-3]) ./ 2).^2)))
    #    .+
    #    (((rho .* ((H[2:end-2,4:end] .+ H[3:end-1,4:end]) ./ 2) .* ((v[2:end-2,4:end-1] .+ v[2:end-2,5:end] .+ v[3:end-1,4:end-1] .+ v[3:end-1,5:end]) ./ 4).^2) .+ (0.5 .* rho .* g .* ((H[2:end-2,4:end] .+ H[3:end-1,4:end]) ./ 2).^2)) .-
    #    ((rho .* ((H[2:end-2,3:end-1] .+ H[3:end-1,3:end-1]) ./ 2) .* ((v[2:end-2,3:end-2] .+ v[2:end-2,4:end-1] .+ v[3:end-1,3:end-2] .+ v[3:end-1,4:end-1]) ./ 4).^2) .+ (0.5 .* rho .* g .* ((H[2:end-2,3:end-1] .+ H[3:end-1,3:end-1]) ./ 2).^2)))
    #) ./ 2
    #)./dy

            if (ix < nx-2) && (iy < ny-2)
            dtv_y[ix,iy] = ((
                (((rho .* ((H[ix+1,iy+1] .+ H[ix+2,iy+1]) ./ 2) .* ((v[ix+1,iy+1] .+ v[ix+1,iy+2] .+ v[ix+2,iy+1] .+ v[ix+2,iy+2]) ./ 4).^2) .+ (0.5 .* rho .* g .* ((H[ix+1,iy+1] .+ H[ix+2,iy+1]) ./ 2).^2)) .-
                ((rho .* ((H[ix+1,iy] .+ H[ix+2,iy]) ./ 2) .* ((v[ix+1,iy] .+ v[ix+1,iy+1] .+ v[ix+2,iy] .+ v[ix+2,iy+1]) ./ 4).^2) .+ (0.5 .* rho .* g .* ((H[ix+1,iy] .+ H[ix+2,iy]) ./ 2).^2)))
                .+
                (((rho .* ((H[ix+1,iy+3] .+ H[ix+2,iy+3]) ./ 2) .* ((v[ix+1,iy+3] .+ v[ix+1,iy+4] .+ v[ix+2,iy+3] .+ v[ix+2,iy+4]) ./ 4).^2) .+ (0.5 .* rho .* g .* ((H[ix+1,iy+3] .+ H[ix+2,iy+3]) ./ 2).^2)) .-
                ((rho .* ((H[ix+1,iy+2] .+ H[ix+2,iy+2]) ./ 2) .* ((v[ix+1,iy+2] .+ v[ix+1,iy+3] .+ v[ix+2,iy+2] .+ v[ix+2,iy+3]) ./ 4).^2) .+ (0.5 .* rho .* g .* ((H[ix+1,iy+2] .+ H[ix+2, iy+3]) ./ 2).^2)))
            ) ./ 2
            )./dy
            end
    #dtv_x = avNbY(diff(rho .* avY(H) .* avX(avY(u)) .* v[:,2:end-1] , dims=1))[2:end-1,:]./dx
    #dtv_y = avNbY(diff((rho .* avX(H) .* avX(avY(v)).^2) .+ (0.5 .* rho .* g .* avX(H).^2) , dims=2))[2:end-1,:]./dy
    #  avNbX(diff(rho .* avX(H) .* u[2:end-1,:] .* avX(avY(v)), dims=2))[:,2:end-1]./dy


            if (ix > 2) && (ix < nx-1) && (iy > 2) && (iy < ny-1)
            v[ix,iy] = (v[ix,iy-1] + v[ix,iy+1])/2 + (
                        (dt * -(
                        dtv_x[ix-2,iy-1] +
                        dtv_y[ix-2,iy-1] 
                        ) / (rho *  (H[ix-1,iy] + H[ix-1,iy+1] + H[ix,iy] + H[ix,iy+1])/4))

                        +

                        (dt * -(
                        dtv_x[ix-2,iy-2] +
                        dtv_y[ix-2,iy-2]
                        ) / (rho *  (H[ix-1,iy-1] + H[ix-1,iy] + H[ix,iy-1] + H[ix,iy])/4))
                    ) /2
            end


    #u .= u_temp
    #v .= v_temp


    #@show size(avX(H[:,2:end-1]))
    #@show size(u)
    #@show size(diff(rho .* avX(H[:,2:end-1]) .* u[2:end-1,2:end-1], dims=1))
    #@show size(diff(rho .* avX(H[:,2:end-1]) .* u[2:end-1,2:end-1], dims=1))

            if (ix > 2) && (ix < nx-1) && (iy > 2) && (iy < ny-1)

            H[ix, iy] = H[ix,iy] + dt * -(
                    ((
                        ((rho * ((H[ix-1,iy] + H[ix,iy]) / 2) * u[ix,iy]) -
                        (rho * ((H[ix-2,iy] + H[ix-1,iy]) / 2) * u[ix-1,iy]))
                        +
                        ((rho * ((H[ix+1,iy] + H[ix+2,iy]) / 2) * u[ix+2,iy]) -
                        (rho * ((H[ix,iy] + H[ix+1,iy]) / 2) * u[ix+1,iy]))
                    ) / 2
                    ) / dx +
                    ((
                        ((rho * ((H[ix,iy-2] + H[ix,iy]) / 2) * v[ix,iy]) -
                        (rho * ((H[ix,iy-2] + H[ix,iy-1]) / 2) * v[ix,iy-1]))
                        +
                        ((rho * ((H[ix,iy+1] + H[ix,iy+2]) / 2) * v[ix,iy+2]) -
                        (rho * ((H[ix,iy] + H[ix,iy+1]) / 2) * v[ix,iy+1]))
                    ) / 2
                    ) / dy
                ) / rho
            end
        #end
    #end
    return
end

"""
Main function of diffusion solver.
"""
@views function swe(nx; do_visu = false)
    # Physics
    lx, ly = 50.0, 50.0       # domain size
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
    dt = 0.09 * min(dx,dy) / sqrt(maximum(H)*g) 
    nt     = cld(ttot, dt)
    @show dt
    #return
 
    u = @zeros(nx+1, ny)
    v = @zeros(nx, ny+1)
    u_temp = @zeros(nx+1, ny)
    v_temp = @zeros(nx, ny+1)

    dtu_x = @zeros(nx - 3, ny - 3)
    dtu_y = @zeros(nx - 3, ny - 3)
    dtv_x = @zeros(nx - 3, ny - 3)
    dtv_y = @zeros(nx - 3, ny - 3)

    # Time loop
    for it = 1:200

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
        #@show size(avNbX(u[2:end-1,4:end-3]))
        #return
        #@show avNbX(diff((rho .* avY(H) .* avX(avY(u)).^2) .+ (0.5 .* rho .* g .* avY(H).^2) , dims=1))[:,2:end-1]


        @parallel (1:nx, 1:ny) compute_step!(H, u, v, dtu_x, dtu_y, dtv_x, dtv_y, nx, ny, g, rho, dt, dx, dy)
 
        @show it
        #display(heatmap(u_temp'))
        #display(heatmap(v_temp'))
        if it % nout == 0
            #display(heatmap(H'))
            display(surface(xc,yc,H',zlims=(0,30)))
            #update!(scene)
            sleep(0.3)
            
        end
        
        #@show u_temp
        #@show v_temp
        #dt = 0.5 * min(dx,dy) / sqrt(maximum(H)*g)
    end

    return H

end

#diffusion3D(256)