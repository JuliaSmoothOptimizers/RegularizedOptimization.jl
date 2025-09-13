using Documenter, DocumenterCitations 

using RegularizedOptimization

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"))

makedocs(
  modules = [RegularizedOptimization],
  doctest = true,
  # linkcheck = true,
  warnonly = [],
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "RegularizedOptimization.jl",
  pages = [
    "Home" => "index.md", 
    "User guide" => [
      joinpath("guide", "introduction.md"),
      joinpath("guide", "algorithms.md"),
      joinpath("guide", "custom.md")
    ], 
    "Examples" => [
      joinpath("examples", "bpdn.md"),
      joinpath("examples", "fh.md")
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
