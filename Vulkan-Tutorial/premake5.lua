workspace "Vulkan-Tutorial"
    architecture "x86_64"

    configurations
    {
        "Debug",
        "Release"
    }

outputdir = "%{cfg.buildcfg}-%{cfg.architecture}"

librarydir = "D:/Programming/VS19_Lib/"
vulkandir = "C:/Program Files/VulkanSDK/1.1.121.2/"

IncludeDir = {}
IncludeDir["Vulkan"]    = vulkandir .. "include"
IncludeDir["GLM"]       = librarydir .. "glm"
IncludeDir["GLFW"]      = librarydir .. "GLFW/include"

LinksDir = {}
LinksDir["Vulkan"]  = vulkandir .. "Lib"
LinksDir["GLFW"]    = librarydir .. "GLFW/lib"

project "Triangle"
    location "Triangle"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    staticruntime "on"
    systemversion "latest"

    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

    --pchheader "hzpch.h"
    --pchsource "Hazel/src/hzpch.cpp"

    files
    {
        "%{prj.name}/src/**.h",
        "%{prj.name}/src/**.cpp"
    }

    includedirs
    {
        "%{prj.name}/src",
        "%{IncludeDir.Vulkan}",
        "%{IncludeDir.GLM}",
        "%{IncludeDir.GLFW}"
    }

    libdirs
    {
        "%{LinksDir.Vulkan}",
        "%{LinksDir.GLFW}"
    }

    links
    {
        "vulkan-1.lib",
        "glfw3.lib"
    }
    
    filter "configurations:Debug"
		--defines "HZ_DEBUG"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		--defines "HZ_RELEASE"
		runtime "Release"
		optimize "on"
