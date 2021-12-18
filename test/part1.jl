# Testing part 1
using BSON

include("../scripts-part1/part1.jl") # modify to include the correct script

# Add unit and reference tests


# Reference test using ReferenceTests.jl
"Compare all dict entries"
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2; atol = 1e-3) for (v1,v2) in zip(values(d1), values(d2))])
inds = Int.(ceil.(LinRange(1, length(xc), 12)))
d = Dict(:X=> xc[inds], :H=>H[inds, inds, 15])

@testset "Ref-file" begin
    @test_reference "reftest-files/test_1.bson" d by=comp
end

d_dual = Dict(:X=> xc_g_2[inds], :H=>H2[inds, inds, 15])

@testset "Ref-file diffusion3D dualtime" begin
    @test_reference "reftest-files/test_1.bson" d_dual by=comp
end
