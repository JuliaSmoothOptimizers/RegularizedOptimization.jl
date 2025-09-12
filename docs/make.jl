using Documenter, RegularizedOptimization

makedocs(
  modules = [RegularizedOptimization],
  doctest = true,
  # linkcheck = true,
  strict = true,
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
      joinpath("examples", "bpdn.md")
      joinpath("examples", "fh.md")
    ], 
    "Reference" => "reference.md"
    ],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl.git",
  push_preview = true,
)
