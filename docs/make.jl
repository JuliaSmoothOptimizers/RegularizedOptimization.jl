using RegularizedOptimization
using Documenter

DocMeta.setdocmeta!(
  RegularizedOptimization,
  :DocTestSetup,
  :(using RegularizedOptimization);
  recursive = true,
)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
  file for file in readdir(joinpath(@__DIR__, "src")) if
  file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
  modules = [RegularizedOptimization],
  authors = "Robert Baraldi <rbaraldi@uw.edu> and Dominique Orban <dominique.orban@gmail.com>",
  repo = "https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl/blob/{commit}{path}#{line}",
  sitename = "RegularizedOptimization.jl",
  format = Documenter.HTML(;
    canonical = "https://JuliaSmoothOptimizers.github.io/RegularizedOptimization.jl",
  ),
  pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl")
