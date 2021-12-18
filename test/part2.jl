# Testing part 2

include("../scripts-part2/part2.jl") # modify to include the correct script

# Add unit and reference tests

# Reference test using ReferenceTests.jl
"Compare all dict entries"
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2; atol = 1e-5) for (v1,v2) in zip(values(d1), values(d2))])
d = Dict(:H=>H)

@testset "Ref-file" begin
    @test_reference "reftest-files/swe.bson" d by=comp
end