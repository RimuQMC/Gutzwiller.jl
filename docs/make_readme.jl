using Literate

Literate.markdown(
    joinpath(@__DIR__, "README.jl");
    flavor=Literate.CommonMarkFlavor(), execute=true,
)
mv(joinpath(@__DIR__, "README.md"), joinpath(@__DIR__, "../README.md"); force=true)
