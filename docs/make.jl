using Documenter
using Gutzwiller

makedocs(;
    modules=[Gutzwiller],
    sitename="Gutzwiller.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
    checkdocs=:exports,
    doctest=false, # doctests are run as part of the test suite, see test/doctests.jl
    warnonly=[:cross_references], # some docstrings reference external Rimu.jl symbols
)

deploydocs(;
    repo="github.com/RimuQMC/Gutzwiller.jl.git",
    devbranch="master",
    push_preview=true,
)
