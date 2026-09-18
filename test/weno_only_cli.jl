# Lightweight driver checks: no GPU, downloaded inputs, or package loading required.
using Test

driver = joinpath(@__DIR__, "..", "scripts", "run_case.jl")
source = read(driver, String)
closure_block = match(r"(?s)if haskey\(opts, \"closure\"\).*?\nend", source)
isnothing(closure_block) && error("Closure option block not found in driver")
closure_expression = Meta.parse(closure_block.match)

@testset "Explicit LES closure option" begin
    for (choice, expected) in (("none", nothing), ("smagorinsky_lilly", :smagorinsky_lilly))
        global opts = Dict("closure" => choice)
        global kw = Dict{Symbol, Any}()
        Core.eval(@__MODULE__, closure_expression)
        @test kw[:closure] === expected
    end
    global opts = Dict{String, String}()
    global kw = Dict{Symbol, Any}()
    Core.eval(@__MODULE__, closure_expression)
    @test !haskey(kw, :closure) # Preserve the preset's default when omitted.
    global opts = Dict("closure" => "invalid")
    @test_throws ArgumentError Core.eval(@__MODULE__, closure_expression)
    launcher = joinpath(@__DIR__, "..", "scripts", "weno_only", "run_member.jl")
    @test Meta.parseall(read(launcher, String)) isa Expr
end
