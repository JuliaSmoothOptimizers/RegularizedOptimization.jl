include("benchmark-nnmf.jl")

include("benchmark-svm.jl")

using LaTeXStrings
all_data = vcat(data_svm, data_nnmf)

table_str = pretty_table(String, all_data;
        header = ["Method", "Status", L"$t$($s$)", L"$\#f$", L"$\#\nabla f$", L"$\#prox$", "Objective"],
        backend = Val(:latex),
        alignment = [:l, :c, :r, :r, :r, :r, :r],
    )

open("Benchmark.tex", "w") do io
    write(io, table_str)
end