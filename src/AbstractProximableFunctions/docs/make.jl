using Documenter, LinearAlgebra, AbstractProximableFunctions

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

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
    modules = [AbstractProximableFunctions],
    sitename = "AbstractProximableFunctions.jl",
    authors = "Gabrio Rizzuti",
    format = format,
    checkdocs = :exports,
    pages = PAGES
)

deploydocs(
    repo = "github.com/grizzuti/AbstractProximableFunctions.jl.git",
)