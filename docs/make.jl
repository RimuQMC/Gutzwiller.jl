using Documenter
using DocumenterInterLinks
using Gutzwiller
using Literate

Literate.markdown(
    joinpath(@__DIR__, "README.jl"), joinpath(@__DIR__, "src", "generated");
    name="guide", flavor=Literate.CommonMarkFlavor(), execute=true,
)

links = InterLinks(
    "Rimu" => "https://rimuqmc.github.io/Rimu.jl/stable/",
)

makedocs(;
    modules=[Gutzwiller],
    sitename="Gutzwiller.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Home" => "index.md",
        "Usage Guide" => "generated/guide.md",
        "API" => "api.md",
    ],
    checkdocs=:exports,
    doctest=false, # doctests are run as part of the test suite, see test/doctests.jl
    plugins=[links],
)

deploydocs(;
    repo="github.com/RimuQMC/Gutzwiller.jl.git",
    devbranch="master",
    push_preview=true,
)
