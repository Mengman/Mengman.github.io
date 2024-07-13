---
layout: post
title: 【学习 cmake step3】 设置编译参数
date: 2024-07-13 00:00:00 +0800
categories: C++ cmake
typora-root-url: ..
---

目标：使用更加现代的方法为 MathFunctions library 设置编译参数

[Step 3: Adding Usage Requirements for a Library](https://cmake.org/cmake/help/latest/guide/tutorial/Adding%20Usage%20Requirements%20for%20a%20Library.html)



## 源码实现

Step3/CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.10)
project(Tutorial VERSION 1.0)

#TODO 4: 使用 library tutorial_compiler_flags 来替换原来的 c++ 标准声明
# 老方法
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(tutorial_compiler_flags INTERFACE)
target_compile_features(tutorial_compiler_flags INTERFACE cxx_std_11)

comfigure_file(TutorialConfig.h.in TutorialConfig.h)

#TODO 2: 删除 EXTRA_INCLUDES list
add_subdirectory(MathFunctions)
# list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")

add_executable(Tutorial tutorial.cxx)

#TODO 5: 添加 tutorial_compiler_flags 到项目的 link 中
target_link_libraries(Tutorial PUBLIC
					  MathFunctions
					  tutorial_compiler_flags)

#TODO 3: 移除子项目的 include 路径
target_include_directories(Tutorial PUBLIC
						  "${PROJECT_BINARY_DIR}"
						  #${EXTRA_INCLUDES}
					)

```

 Step3/MathFunctions/CMakeLists.txt
```cmake
add_library(MathFunctions MathFunctions.cxx)

#TODO 1: 声明任何依赖 MathFunnctions 库的项目都需要 include 当前源代码目录
target_include_directories(MathFunctions INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

option(USE_MYMATH "Use tutorial provided math implementation" ON)
if (USE_MYMATH)
  target_compile_definitions(MathFunctions PRIVATE "USE_MYMATH")
  add_library(SqrtLibrary STATIC
              mysqrt.cxx
              )
  
  # TODO 6: Link SqrtLibrary to tutorial_compiler_flags
  target_link_libraries(SqrtLibrary PUBLIC tutorial_compiler_flags)
  target_link_libraries(MathFunctions PRIVATE SqrtLibrary)
endif()
  
# TODO 7: Link MathFunctions to tutorial_compiler_flags
target_link_directories(MathFunctions PUBLIC tutorial_compiler_flags)
```

总结一下。第一，通过在 MathFunctions 库用 `target_include_directories` 命令声明了自己的 include 依赖，从而使得每个依赖 MathFunctions 库的项目都会自动 include 声明的路径；第二， 结合声明 `INTERFACE` 模式的 library `tutorial_compiler_flags` 并且设置它的编译功能 `target_compile_features` 来替换，原来基于 `set` 命令方式设置 C++ 标准，它的好处是子项目直接添加 `tutorial_compiler_flags` 依赖就好，避免重复设置 `CMAKE_CXX_STANDARD` 变量。

## interface 作用域

### target_include_directories
在 step1 中简单介绍过 interface 作用域的作用是 “添加的目录将作为接口包含目录,对于依赖于该目标的其他目标可见”。这里再深入介绍一下 `target_include_directories` 在 `INTERFACE` 作用域下的表现。

当 `target_include_directories` 命令将作用域设置为 `INTERFACE`形式,那么被添加的包含**目录不会被该目标本身使用**, 而是被传递给链接到该目标的其他依赖目标使用。

例如,如果有一个静态库 `mylib`依赖于第三方库 `thirdparty`,在编译`mylib`时需要包含`thirdparty`的头文件路径。我们可以这样写:
```cmake
add_library(mylib ....)
target_include_directories(mylib INTERFACE /path/to/thirdparty/include)
```
这样,当其他目标链接`mylib`时,`/path/to/thirdparty/include`这个包含目录也会被自动传递给它们使用。
总之,`target_include_directories`的`INTERFACE`作用域为处理包含目录依赖传递、模块化构建等提供了支持,值得在 CMakeLists.txt 中合理使用。
### add_library
在CMake中，当使用`add_library`命令时，如果将作用域设置为`INTERFACE`，则不会生成实际的库文件，而只是定义一个接口目标(Interface Target)。接口目标的作用是为依赖于它的其他目标提供使用信息,比如包含目录、链接库、编译定义等。
使用接口目标的主要目的有以下几点:
1. **模块化构建**
   接口目标为将某些构建设置(如编译选项、依赖项等)集中到一个地方提供了一种方式,实现了构建系统的模块化。其他目标只需链接接口目标,就能获得相应的构建设置。
2. **依赖传递**
   通过`target_link_libraries`将接口目标链接到其他目标时,接口目标暴露的所有使用信息(包括从其他依赖目标继承的)都会传递给链接的目标。这样可以自动处理依赖的依赖。
3. **跨平台支持**
   接口目标可以根据不同平台设置不同的构建选项,从而实现跨平台支持。其他目标只需链接接口目标,无需关心具体的平台细节。
4. **可重用性**
   接口目标本身不生成实际文件,所以易于在不同项目之间重用。
5. **依赖可见性**
   将依赖项集中在接口目标中,可以提高构建系统的可见性和可理解性。
   一个典型的使用接口目标的例子是,为某个第三方库创建一个接口库,将该库所需的所有构建信息集中在接口目标中,然后项目中其他目标只需链接这个接口目标即可。

## target_compile_features
`target_compile_features` 命令, 用于为目标(可执行程序或库)指定需要启用的编译器特性。帮助 target 利用编译器提供的最新语言和硬件特性,以编写更高效、更现代的代码。

以下是`target_compile_features`命令的基本用法:
```cmake
target_compile_features(<target>
  <PRIVATE|PUBLIC|INTERFACE> <feature> ...
  [PRIVATE|PUBLIC|INTERFACE] <feature> ...
)
```
- `<target>`: 指定要启用编译器特性的目标,可以是可执行程序或库。
- `<PRIVATE|PUBLIC|INTERFACE>`: 指定编译器特性的作用范围。
  - `PRIVATE`: 仅应用于目标本身。
  - `PUBLIC`: 应用于目标本身和依赖于该目标的其他目标。
  - `INTERFACE`: 仅应用于依赖于该目标的其他目标。
- `<feature>`: 指定要启用的编译器特性,例如`cxx_std_11`用于启用C++11标准。

示例:
```cmake
add_executable(myapp main.cpp)

# 为 myapp 启用 C++11 标准
target_compile_features(myapp PUBLIC cxx_std_11)

# 为 myapp 启用 C++14 标准和 OpenMP
target_compile_features(myapp PRIVATE cxx_std_14 openmp)
```
在上面的示例中,对于`myapp`目标,我们启用了C++11标准(对于`myapp`本身和依赖于它的其他目标),并且还私有地启用了C++14标准和OpenMP特性(仅对`myapp`本身)。

一些常见的`target_compile_features`可设置的特性:

| 特性         | 特性关键字                                                   | 描述                                         |
| ------------ | ------------------------------------------------------------ | -------------------------------------------- |
| C++标准特性  | `cxx_std_98`, `cxx_std_11`, `cxx_std_14`, `cxx_std_17`, `cxx_std_20`, `cxx_std_23` | 启用对应版本的C++标准支持                    |
| OpenMP特性   | `openmp`                                                     | 启用OpenMP并行编程支持                       |
| CUDA特性     | `cuda_stdpar`, `cuda_std_11`, `cuda_std_14`, `cuda_std_17`   | 启用对应版本的CUDA标准并行算法支持           |
| 综合特性     | `stdpar_latest`, `cuda_latest`                               | 启用最新的标准并行算法支持(包括C++和CUDA)    |
| 硬件特性     | `sse`, `sse2`, `avx`, `avx2`, `fma`                          | 启用对应的SIMD指令集支持                     |
| 警告选项     | `warn_uninitialized`, `warn_nullptr`                         | 启用相关警告选项                             |
| 调试特性     | `address_sanitizer`, `thread_sanitizer`, `undefined_behavior_sanitizer` | 启用相应的内存/线程/未定义行为检测器         |
| 其他语言特性 | `c_std_99`, `c_std_11`, `objc_std_11`                        | 分别启用对应版本的C语言和Objective-C语言支持 |