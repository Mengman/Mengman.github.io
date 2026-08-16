---
title: 【学习 cmake step4】 Generator Expression
date: 2026-08-16T02:24:31.588Z
tags: [C++, cmake]
categories: cpp
---

目标: 学习使用 cmake 中的 **Generator Expression** 语法。
Generator expressions 是 CMake 中一种强大的功能,它允许用户根据生成系统的状态有条件地设置变量或选项。这种表达式在构建时进行计算和展开,因此可以根据构建配置的不同动态地设置值。

## 源码实现
使用 **Generator Expression**(GE) 根据不同的编译器，为 target 添加不同的编译参数。

Step4/CMakeLists.txt
```cmake
# TODO 1: "Generator expressions" 要求 cmake版本 >= 3.15
cmake_minimum_required(VERSION 3.15)

add_library(Tutorial VERSION 1.0)
add_library(tutorial_compiler_flags INTERFACE)
target_compile_features(tutorial_compiler_flags INTERFACE cxx_std_11)

# TODO 2: 创建辅助变量用来判断编译时候使用的编译器类型
# gcc_like_cxx=True：当编程语言是CXX，并且编译器是以下编译中的一种 [ARMClang,AppleClang,Clang,GNU,LCC] 
set(gcc_like_cxx "$<COMPILE_LAND_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
# msvc_cxx=True: 当编程语言是CXX，并且编译器是 MSVC
set(msvc_cxx "$<COMPILE_LAND_AND_ID:CXX,MSVC>")

# TODO 3: 为编译器添加 warning flags
# gcc like 编译器编译 flag '-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused'
# msvc 编译器 flag '-W3'
# TODO 4: 添加 'BUILD_INTERFACE' 确保编译参数只用于当前项目
target_compile_options(tutorial_compiler_flags INTERFACE
						"$<$<gcc_like_cxx>:$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
						"$<$<msvc_cxx>:$<BUILD_INTERFACE:-W3>"
						)

configure_file(TutorialConfig.h.in TutorialConfig.h)
add_subdirectory(MathFunctions)
add_executable(Tutorial tutorial.cxx)
target_link_libraries(Tutorial PUBLIC MathFunctions tutorial_compiler_flags)
target_include_directories(Tutorial PUBLIC
							"${PROJECT_BINARY_DIR}"
							)

```

新增代码的作用，使用 GE 检查编译器的类型，然后更加不同的编译器设置不同的编译器参数 flags。

## Generator Expression 基础语法
GE 的形式是 `$<...>` , 它是在 CMake 的配置阶段进行求值和计算的,也就是在生成构建系统的文件之前。
### 表达式解析
GE 通常都是放在 CMake 命令中，作为命令参数的一部分。 前面说了 GE 是在 CMake 配置阶段进行求值和计算的。在具体执行解析时候，首先将命令行分割成各个参数,然后再解析每个参数中的 generator expressions。

如果一个 generator expression 包含空格、新行、分号或其他可能被解释为命令参数分隔符的字符,那么在传递给命令时,整个表达式应该用引号包裹起来。如果不这样做,表达式可能会被分割,从而无法被正确识别为一个 generator expression。

当使用 `add_custom_command()` 或 `add_custom_target()` 命令时,应该使用 `VERBATIM` 和 `COMMAND_EXPAND_LISTS` 选项,以获得稳健的参数分割和引用。
- `VERBATIM` 选项告诉 CMake 按原样传递所有参数,不对任何值做进一步的解析或修改。
- `COMMAND_EXPAND_LISTS` 将列表扩展为单独的参数,通常与 `VERBATIM` 一起使用。
使用这两个选项可以确保 generator expressions 传递给命令时不会被意外分割或修改。

### 条件表达式

#### `$<condition:true_string>` 
根据给定条件 `condition` 的真假返回字符串， 如果 `condition` 为 1，那么就返回字符串 `true_string`；否则返回空字符串。
```cmake
# 如果编程语言是 CXX，那么就返回 '-std=c++11'
$<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
```

#### `$<IF:condition,true_string,false_string>` 
如果 `condition` 为 1， 返回字符串 `true_string`; 否则返回 `false_string`

#### `$<BOOL:string>`
将字符串转换为 bool 值(0, 1)。
如果字符串满足一下情况返回 0:
* `string` 是空字符串
* `string` 的内容等于(不区分大小写)， "0", "FALSE", "OFF", "N", "NO", "IGNORE", "NOTFOUND"
* `string` 以 '-NOTFOUND' 为结尾 (不区分大小写)

以上情况之外都会返回 1。

### 逻辑运算
#### `$<AND:conditions>`
逻辑于， `conditions` 是多个用逗号分隔的布尔表达式，表达式的结果为`conditions` 的逻辑于结果。
```cmake
$<AND:$<CONFIG:Debug>,$<PLATFORM_ID:Windows>>
```

#### `$<OR:conditions>`
逻辑或， `conditions` 是多个用逗号分隔的布尔表达式，表达式的结果为`conditions` 的逻辑或结果。

#### `$<NOT:condition>`
逻辑非， `condition` 是一个布尔表达式，表达式的结果为`condition` 的逻辑非结果。

### 比较表达式

#### 数值比较 `$<EQUAL:value1,value2>`
如果 value1 value2 都是数字并且值相等返回 1，否则返回 0

#### 版本比较

`$<VERSION_LESS:v1,v2>` 
如果 v1 版本比 v2 小返回 1，否则返回 0

`$<VERSION_GREATER:v1,v2>`
如果 v1 版本比 v2 大返回 1，否则返回 0


### 字符串表达式

#### 字符串比较
`$<STREQUAL:string1,string2>`
比较字符串是否相等

`$<STRLESS:string1,string2>`
`$<STRGREATER:string1,string2>`
`$<STRLESS_EQUAL:string1,string2>`
`$<STRGREATER_EQUAL:string1,string2>`
按照字典序比较字符串的大小


更多 GE 表达式参见 [文档](https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html)

