cmake_minimum_required(VERSION 3.13)
project(LearningVulkan)


find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)


set(LearningVulkan_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(VkRenderer_SRC ${LearningVulkan_SRC_DIR}/core.hpp
                   ${LearningVulkan_SRC_DIR}/Window.hpp
                   ${LearningVulkan_SRC_DIR}/Window.cpp
                   ${LearningVulkan_SRC_DIR}/VkBackend.hpp
                   ${LearningVulkan_SRC_DIR}/VkBackend.cpp)

set(LearningVulkan_SRC ${LearningVulkan_SRC_DIR}/main.cpp)

add_executable(LearningVulkan ${LearningVulkan_SRC} ${VkRenderer_SRC})
target_include_directories(LearningVulkan PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(LearningVulkan ${Vulkan_LIBRARIES} glfw glm)

# THIS SHIT DOESN'T WORK
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(LearningVulkan PRIVATE "/std:c++latest")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest")
else()
    target_compile_features(LearningVulkan PRIVATE cxx_std_20)
endif()
