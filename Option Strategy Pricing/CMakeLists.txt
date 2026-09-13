cmake_minimum_required(VERSION 3.16)

project(EuropeanOption LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(european_option_lib EuropeanOption.cpp)
target_include_directories(european_option_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(strategy_analyzer TestEuropeanOption.cpp)
target_link_libraries(strategy_analyzer PRIVATE european_option_lib)

add_executable(option_regression OptionRegression.cpp)
target_link_libraries(option_regression PRIVATE european_option_lib)

enable_testing()
add_test(NAME option_regression COMMAND option_regression)
