using Documenter, AbstractProximableFunctions, FastSolversForWeightedTV, RetrospectiveMotionCorrectionMRI

const RealOrComplex{T<:Real} = Union{T, Complex{T}}

format = Documenter.HTML()

Introduction = "Introduction" => "index.md"
Installation = "Installation" => "installation.md"
GettingStarted = "Getting started" => "examples.md"
MainFunctions = "Main functions" => "functions.md"

PAGES = [
    Introduction,
    Installation,
    GettingStarted,
    MainFunctions
    ]

makedocs(
    modules = [AbstractProximableFunctions, FastSolversForWeightedTV, RetrospectiveMotionCorrectionMRI],
    sitename = "RetrospectiveMotionCorrectionMRI.jl",
    authors = "Gabrio Rizzuti",
    format = format,
    checkdocs = :exports,
    pages = PAGES
)

deploydocs(
    repo = "github.com/grizzuti/RetrospectiveMotionCorrectionMRI.jl.git",
)