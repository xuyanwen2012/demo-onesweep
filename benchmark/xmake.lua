add_requires("benchmark")

-- for each .cpp files in current directory
for _, file in ipairs(os.files("*.cu")) do
    local name = "bench_" .. path.basename(file)
    target(name)
        set_kind("binary")
        add_includedirs("../include")
        add_files(file, "../src/init.cu", "../src/one_sweep.cu")
        add_cugencodes("native")
        add_packages("benchmark")
end

