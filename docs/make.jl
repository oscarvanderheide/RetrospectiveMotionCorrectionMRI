using Documenter, RetrospectiveMotionCorrectionMRI

const RealOrComplex{T<:Real} = Union{T, Complex{T}}

format = Documenter.HTML()

Introduction = "Introduction" => "index.md"
Installation = "Installation" => "installation.md"
Theory = "Rigid motion correction for MRI" => "theory.md"
GettingStarted = "Getting started" => "examples.md"
MainFunctions = "Main functions" => "functions.md"

PAGES = [
    Introduction,
    Installation,
    Theory,
    GettingStarted,
    MainFunctions
    ]

makedocs(
    modules = [RetrospectiveMotionCorrectionMRI],
    sitename = "RetrospectiveMotionCorrectionMRI.jl",
    authors = "Gabrio Rizzuti",
    format = format,
    checkdocs = :exports,
    warnonly = [:missing_docs, :cross_references],
    pages = PAGES
)

deploydocs(
    repo = "github.com/oscarvanderheide/RetrospectiveMotionCorrectionMRI.git",
)