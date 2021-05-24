outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

workspace "neural_network"
    startproject "digit_recognition"

    configurations { "Debug", "Release", "Dist" }

    -- when "IDE" defined, program reads form cwd instead of execution location
    filter "configurations:Debug"
        defines { "DEBUG", "IDE" }
        symbols "On"
    filter "configurations:Release"
        defines { "RELEASE", "IDE" }
        symbols "On"
        optimize "On"
    filter "configurations:Dist"
        defines { "DIST" }
        optimize "On"
    
    filter "system:linux"
        defines { "LINUX" }
    filter "system:windows"
        defines { "WINDOWS" }

    -- cwd
    debugdir ""

include "digit_recognition"
include "utils"
include "vendor/armadillo"

