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
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/UW-AMO/RegularizedOptimization.jl.git",
  push_preview = true
)

