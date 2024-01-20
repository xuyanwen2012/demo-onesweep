add_rules("mode.debug", "mode.release")

set_languages("c++17")
set_optimize("fastest")
set_warnings("all")

target("demo-onesweep")
    set_kind("binary")
    add_includedirs("include")
    add_headerfiles("include/*.cuh", "include/**/*.hpp")
    add_files("src/*.cu")
    add_cugencodes("native")
