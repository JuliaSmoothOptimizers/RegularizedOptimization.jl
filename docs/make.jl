using Documenter, DocumenterCitations 

using RegularizedOptimization

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"))

makedocs(
  modules = [RegularizedOptimization],
  doctest = true,
  # linkcheck = true,
  warnonly = false,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "RegularizedOptimization.jl",
  pages = [
    "Home" => "index.md", 
    "Algorithms" => "algorithms.md",
    "Examples" => [
      joinpath("examples", "basic.md"),
      joinpath("examples", "ls.md"),
    ], 
    "Reference" => "reference.md",
    "Bibliography" => "bibliography.md"
    ],
    plugins = [bib],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl.git",
  push_preview = true,
)
