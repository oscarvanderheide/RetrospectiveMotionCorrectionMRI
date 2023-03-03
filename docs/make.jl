using Documenter, AbstractProximableFunctions, FastSolversForWeightedTV, RetrospectiveMotionCorrectionMRI

const RealOrComplex{T<:Real} = Union{T, Complex{T}}

format = Documenter.HTML()

Introduction = "Introduction" => "index.md"
Installation = "Installation" => "installation.md"
GettingStarted = "Getting started" => "examples.md"
ParameterEstimation = "Rigid motion parameter estimation" => "parameter_estimation.md"
ImageReconstruction = "Image reconstruction" => "image_reconstruction.md"
MainFunctions = "Main functions" => "functions.md"

PAGES = [
    Introduction,
    Installation,
    GettingStarted,
    ImageReconstruction,
    ParameterEstimation,
    MainFunctions
    ]

makedocs(
    modules = [RetrospectiveMotionCorrectionMRI],
    sitename = "RetrospectiveMotionCorrectionMRI.jl",
    authors = "Gabrio Rizzuti",
    format = format,
    checkdocs = :exports,
    pages = PAGES
)

deploydocs(
    repo = "github.com/grizzuti/RetrospectiveMotionCorrectionMRI.jl.git",
)