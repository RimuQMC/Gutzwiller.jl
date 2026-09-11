using Rimu
using Documenter
using Gutzwiller
using Optim

DocMeta.setdocmeta!(
    Gutzwiller,
    :DocTestSetup,
    :(using Gutzwiller; using Rimu; using Optim; using NLSolversBase);
    recursive=true,
)

doctest(Gutzwiller; doctestfilters=[r"(\d*)\.(\d{4})\d+" => s"\1.\2"], manual=false)
