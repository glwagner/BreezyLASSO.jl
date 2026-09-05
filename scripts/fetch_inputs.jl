# Download the public Covert et al. (2022) SAM input files and record provenance.
using SHA

const REPOS = [
    (name = "covert2022_bulk", url = "https://github.com/dmechem/ENA_variability_LES_bulk_paper.git",
     files = ["snd" => "snd", "lsf_time_varying" => "lsf", "sfc_time_varying" => "sfc"]),
    (name = "covert2022_bin", url = "https://github.com/dmechem/ENA_variability_LES_bin_paper.git",
     files = ["snd" => "snd", "lsf" => "lsf", "sfc" => "sfc", "prm" => "prm"]),
]

data_dir = joinpath(@__DIR__, "..", "data")
mkpath(data_dir)
manifest = open(joinpath(data_dir, "MANIFEST.txt"), "w")


function now_utc()
    return Base.Libc.strftime("%Y-%m-%dT%H:%M:%SZ", time())
end

for repo in REPOS
    target = joinpath(data_dir, repo.name)
    tmp = joinpath(data_dir, repo.name * "_git")
    rm(tmp; force=true, recursive=true)
    run(`git clone -q $(repo.url) $tmp`)
    commit = strip(read(`git -C $tmp rev-parse HEAD`, String))
    mkpath(target)
    println(manifest, "\n[", repo.name, "]\nurl = \"", repo.url, "\"\ncommit = \"", commit, "\"")
    for (src, dst) in repo.files
        cp(joinpath(tmp, src), joinpath(target, dst); force=true)
        digest = bytes2hex(open(sha256, joinpath(target, dst)))
        println(manifest, dst, " = \"", src, "\"  # sha256 ", digest)
    end
    rm(tmp; force=true, recursive=true)
end
close(manifest)
println("inputs written to ", abspath(data_dir))
