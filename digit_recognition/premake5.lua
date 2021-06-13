project "digit_recognition"
    language "C++"
    architecture "x86_64"
    cppdialect "C++17"
    kind "ConsoleApp"

    warnings "Extra"

    -- like in #include
    pchheader "pch.h"
    -- relative path
    pchsource "src/pch.cpp"

    location ""
    targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
    objdir ("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")

    files {
        "src/**.h",
        "src/**.cpp",
    }

    -- for #include with ""
    includedirs {
        "src",
        "%{wks.location}/utils/src",
    }

    -- for #include with <>
    sysincludedirs {
        "%{wks.location}/vendor/armadillo/include",
        "%{wks.location}/vendor/json",
    }

    links {
        "utils",
        "armadillo",
        "lapack",
        "blas",
    }

    filter "not system:windows"
    links {
        "pthread",
    }
