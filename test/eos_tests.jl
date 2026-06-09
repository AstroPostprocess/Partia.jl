######################################################################################

#  Test: Equation of State — Thermodynamic Helpers
#  What this file tests
#  Validates equation-of-state helper utilities:
#  1. Sound speed
#     • Adiabatic, isothermal, and locally isothermal modes.
#     • Invalid-input NaN behavior and mixed numeric-type promotion.
#  2. Pressure
#     • Adiabatic, isothermal, and locally isothermal modes.
#     • Invalid density, internal energy, radius, and adiabatic-index handling.
#  3. Temperature
#     • SI, CGS, and Galactic unit dispatch for supported EOS modes.
#     • Invalid sound-speed, radius, and adiabatic-index handling.
#  4. Enthalpy
#     • Adiabatic, isothermal, and locally isothermal modes.
#     • Invalid thermodynamic input handling.

######################################################################################
using Test
using Partia

# ============================== Test body =================================== #

# ── 1. Sound speed — equation-of-state modes ─────────────────────────── #

@testset "SoundSpeed -- Adiabatic" begin
    γ = 5.0 / 3.0
    u = 2.0
    cs = SoundSpeed(Adiabatic, u, γ)
    @test cs ≈ sqrt(γ * (γ - 1) * u)

    cs_mixed = SoundSpeed(Adiabatic, 2.0f0, γ)
    @test cs_mixed isa Float64

    @test isnan(SoundSpeed(Adiabatic, -1.0, γ))
    @test isnan(SoundSpeed(Adiabatic, 2.0, 0.5))
end

@testset "SoundSpeed -- Isothermal" begin
    cs0 = 0.3
    @test SoundSpeed(Isothermal, cs0) ≈ cs0
end

@testset "SoundSpeed -- LocallyIsothermal" begin
    cs0 = 1.0
    r = 2.0
    q = 0.25
    cs = SoundSpeed(LocallyIsothermal, r, cs0, q)
    @test cs ≈ cs0 * r^(-q)

    @test isnan(SoundSpeed(LocallyIsothermal, -1.0, cs0, q))
    @test isnan(SoundSpeed(LocallyIsothermal, 0.0, cs0, q))
end

# ── 2. Pressure — equation-of-state modes ────────────────────────────── #

@testset "Pressure -- Adiabatic" begin
    γ = 5.0 / 3.0
    ρ = 1.5
    u = 2.0
    P = Pressure(Adiabatic, ρ, u, γ)
    @test P ≈ (γ - 1) * ρ * u

    @test isnan(Pressure(Adiabatic, -1.0, u, γ))
    @test isnan(Pressure(Adiabatic, ρ, -1.0, γ))
    @test isnan(Pressure(Adiabatic, ρ, u, 0.5))
end

@testset "Pressure -- Isothermal" begin
    ρ = 1.0
    cs = 0.3
    @test Pressure(Isothermal, ρ, cs) ≈ ρ * cs^2

    @test isnan(Pressure(Isothermal, -1.0, cs))
end

@testset "Pressure -- LocallyIsothermal" begin
    ρ = 1.0
    r = 2.0
    cs0 = 1.0
    q = 0.25
    P = Pressure(LocallyIsothermal, ρ, r, cs0, q)
    @test P ≈ ρ * (cs0 * r^(-q))^2

    @test isnan(Pressure(LocallyIsothermal, -1.0, r, cs0, q))
    @test isnan(Pressure(LocallyIsothermal, ρ, 0.0, cs0, q))
end

# ── 3. Temperature — equation-of-state modes ─────────────────────────── #

@testset "Temperature -- Adiabatic" begin
    γ = 5.0 / 3.0
    u = 2.0
    μ = 2.3
    @test Temperature(Adiabatic, SIUnit, u, γ, μ) ≈ 0.00012114751277768644 * μ * (γ - 1.0) * u
    @test Temperature(Adiabatic, CGSUnit, u, γ, μ) ≈ 1.2114751277768644e-8 * μ * (γ - 1.0) * u
    @test Temperature(Adiabatic, GalacticUnit, 2.0f0, γ, μ) isa Float64
    @test isnan(Temperature(Adiabatic, SIUnit, u, 0.5, μ))
end

@testset "Temperature -- Isothermal" begin
    cs = 0.3
    μ = 2.3
    @test Temperature(Isothermal, SIUnit, cs, μ) ≈ 0.00012114751277768644 * μ * cs^2
    @test Temperature(Isothermal, GalacticUnit, cs, μ) ≈ 121.14751277768644 * μ * cs^2
    @test isnan(Temperature(Isothermal, CGSUnit, -cs, μ))
end

@testset "Temperature -- LocallyIsothermal" begin
    r = 2.0
    cs0 = 1.0
    q = 0.25
    μ = 2.3
    cs = cs0 * r^(-q)
    @test Temperature(LocallyIsothermal, SIUnit, r, cs0, q, μ) ≈ 0.00012114751277768644 * μ * cs^2
    @test Temperature(LocallyIsothermal, CGSUnit, 2.0f0, cs0, q, μ) isa Float64
    @test isnan(Temperature(LocallyIsothermal, GalacticUnit, 0.0, cs0, q, μ))
end

# ── 4. Enthalpy — equation-of-state modes ────────────────────────────── #

@testset "Enthalpy -- Adiabatic" begin
    γ = 5.0 / 3.0
    u = 2.0
    h = Enthalpy(Adiabatic, u, γ)
    @test h ≈ γ * u

    h_mixed = Enthalpy(Adiabatic, 2.0f0, γ)
    @test h_mixed isa Float64

    @test isnan(Enthalpy(Adiabatic, -1.0, γ))
    @test isnan(Enthalpy(Adiabatic, u, 0.5))
end

@testset "Enthalpy -- Isothermal" begin
    u = 2.0
    cs = 0.3
    h = Enthalpy(Isothermal, u, cs)
    @test h ≈ u + cs^2

    h_mixed = Enthalpy(Isothermal, 2.0f0, cs)
    @test h_mixed isa Float64

    @test isnan(Enthalpy(Isothermal, -1.0, cs))
    @test isnan(Enthalpy(Isothermal, u, -cs))
end

@testset "Enthalpy -- LocallyIsothermal" begin
    u = 2.0
    r = 2.0
    cs0 = 1.0
    q = 0.25
    h = Enthalpy(LocallyIsothermal, u, r, cs0, q)
    @test h ≈ u + (cs0 * r^(-q))^2

    h_mixed = Enthalpy(LocallyIsothermal, 2.0f0, r, cs0, q)
    @test h_mixed isa Float64

    @test isnan(Enthalpy(LocallyIsothermal, -1.0, r, cs0, q))
    @test isnan(Enthalpy(LocallyIsothermal, u, 0.0, cs0, q))
    @test isnan(Enthalpy(LocallyIsothermal, u, r, -cs0, q))
end
