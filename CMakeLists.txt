cmake_minimum_required(VERSION 3.13)
project(LearningVulkan)

# For FindGLFW3.cmake
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(glm_DIR $ENV{GLM_ROOT}/cmake/glm) # set GLM_ROOT

find_package(Vulkan REQUIRED)
find_package(GLFW3 3.3 REQUIRED) # set GLFW3_ROOT
find_package(glm REQUIRED)


set(LearningVulkan_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(LearningVulkan_SRC ${LearningVulkan_SRC_DIR}/main.cpp)

add_executable(LearningVulkan ${LearningVulkan_SRC})
target_include_directories(LearningVulkan PRIVATE ${Vulkan_INCLUDE_DIRS} ${GLFW3_INCLUDE_DIR})
target_link_libraries(LearningVulkan ${Vulkan_LIBRARIES} ${GLFW3_LIBRARY} glm::glm)
target_compile_features(LearningVulkan PRIVATE cxx_std_20)