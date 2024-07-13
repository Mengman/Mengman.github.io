---
layout: post
title: 【学习 cmake step1】 创建一个简单的 C++ 项目
date: 2024-01-04 00:00:00 +0800
categories: cmake C++ softwareenginer build
typora-root-url: ..
---



本文是我学习 [cmake 官方教程](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)的笔记，我在官方教程的基础上，对其中的知识点做了展开描述。



# 什么是 cmake

CMake 是一个跨平台的软件构建工具。

它的主要功能包括
* 跨平台：CMake 可以生成适用于 Windows、Linux、macOS 等多种平台的项目文件。
* 灵活：CMake 的配置文件使用一种类似 C++ 的语言，可以灵活地描述软件的构建过程。
* 可扩展：CMake 提供了丰富的 API，可以用于扩展其功能。

CMake 支持构建多种编程语言项目，但是在 C/C++ ， CUDA 项目中使用的最广泛。



## cmake 与编译器的关系

C/C++ 项目有很多中编译器如 Linux/Unix 上最常用的 GCC， Clang， Windows 上的 MSVC



## cmake 与 make 的对比

`CMake` 和 `make` 都是 C/C++ 项目常用的自动化编译工具，帮助实现项目依赖管理、源代码编译与依赖库链接。 `CMake`  通过编写 `CMakeLists.txt` 文件来配置项目构建过程；`make` 通过编写 `Makefiles` 来配置项目构建过程。

它们之间又有明显的区别:

| 特征     | `CMake`                                                      | `make`                                                       |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 功能     | 它是一个强大的元构建系统， 它根据项目配置来生成构建脚本(如 Makefiles) | 一个根据 Makefiles 来执行命令行指令的构建工具                |
| 抽象程度 | 高，可以编写与平台独立的 CMakeLists.txt 文件， cmake 会根据 CMakelists.txt 和具体平台、构建工具与编译器生成具体的编译脚本 | 低，Makefiles 内容与特定平台、编译器绑定                     |
| 灵活性   | 高，`CMake` 支持生成 `make` `ninja` 等构建项目的配置文件，并且支持通过配置变量来支持不同的操作系统和编译器 | 低，对于跨平台、多编译支持较为困难                           |
| 复杂性   | 高，`CMake` 的功能强大，学习曲线较高                         | 低，`Makefiles` 的语法简单                                   |
| 输出     | 生成构建脚本（如 Makefiles）                                 | 编译完成的结果                                               |
| 适合项目 | 大型跨平台 C/C++、CUDA 项目                                  | 小型项目、特定平台的项目，或者构建过程特别简单的项目（如 Golang 项目） |



# 创建一个简单的 C++ 项目



## 源码实现

[Step 1: A Basic Starting Point](https://cmake.org/cmake/help/latest/guide/tutorial/A%20Basic%20Starting%20Point.html)

构建一个简单的可执行项目
目标：

1. 设置 cmake 版本要求
2. 设置项目名称
3. 设置项目可执行入口
4. 设置 C++ 语言版本
5. 生成一个配置头文件

CMakeLists.txt
```cmake
# 设置 cmake 版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(Tutorial VERSION 1.0)

# 设置 C++ 版本为 11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

configure_file(TutorialConfig.h.in TutorialConfig.h)

# 设置项目入口
add_executable(Tutorial tutorial.cxx)

target_include_directories(
	Tutorial PUBLIC
	"${PROJECT_BINARY_DIR}"
)
```

TutorialConfig.h.in
```c++
#define Tutorial_VERSION_MAJOR @Tutorial_VERSION_MAJOR@
#define Tutorial_VERSION_MINOR @Tutorial_VERSION_MINOR@
```
## 命令讲解

### cmake_minimum_required 命令

**cmake_minimum_required** 是用来设置 cmake 版本要求的它的格式是
`cmake_minimum_required(VERSION <min_version> [FATAL_ERROR])`
* `<min_version>`: 是支持的最小版本号， 如果要设置一个范围可以这么写 `3.10...3.27` 表示支持最老 3.10 到最新 3.27 版本的 cmake。
* `FATAL_ERROR`是可选参数,如果指定了该参数,当检测到CMake的版本低于要求的最小版本时,CMake将终止并报错,而不是发出警告继续运行。

### project 命令
它用来指定项目的名称、版本和支持的编程语言。
```cmake
project(ProjectName 
		VERSION MajorVersion.MinorVersion.PatchVersion 
		LANGUAGES languageName1 languageName2 ...)
```
`project(projectName [LANGUAGES] [languageName1 languageName2 ...])`
- `projectName` 是项目的名称,这是该命令的必需参数。
- `VERSION` 是一个关键字, 用来指定项目的版本号
- `MajorVersion.MinorVersion.PatchVersion ` 带表项目的版本号
- `LANGUAGES` 是一个关键字,用于指定项目支持的编程语言,这是可选参数。
- `languageName1 languageName2 ...` 是一个或多个编程语言的名称,如`C`、`CXX`(C++)、`CUDA`等,这也是可选参数。

使用该命令通常有以下几个目的:
1. **设置项目名称**: 项目名称会用于生成各种文件和目标文件的名称前缀。
2. **启用语言**: 指定项目所使用的编程语言,CMake会自动包含适当的编译器和构建规则。如果没有指定,默认启用`C`和`CXX`语言。
3. **设置编译器选项**: CMake会根据指定的语言自动设置相应的编译器选项和预处理选项。
4. **配置构建系统**: 根据指定的语言,CMake会配置合适的构建系统,如Makefile或Visual Studio项目文件。

### add_executable 命令
`add_executable` 命令是CMake中用于创建可执行目标的命令。它的基本语法如下:
```cmake
add_executable(<name> [WIN32] [MACOSX_BUNDLE]
               [EXCLUDE_FROM_ALL]
               [source1] [source2 ...])
```
- `<name>` 是必需的,用于指定生成的可执行文件的名称。
- `[WIN32]` 是可选的,用于指示在Windows平台上生成一个Windows GUI程序。如果没有指定,默认会生成控制台程序。
- `[MACOSX_BUNDLE]` 是可选的,用于在macOS平台上生成一个应用程序包(bundle)。
- `[EXCLUDE_FROM_ALL]` 是可选的,用于指示该目标不应该被默认构建规则构建,需要手动构建。
- `[source1] [source2 ...]` 是可选的,用于指定构建该可执行文件所需的源文件。如果没有列出源文件,那么CMake会查找与`<name>`同名的源文件。

这个命令会将列出的源文件编译并链接成一个可执行文件。生成的可执行文件的名称和位置取决于本地构建工具的规则。

一些使用示例:

```cmake
# 创建一个简单的可执行文件
add_executable(myapp main.cpp util.cpp)

# 在Windows上创建一个GUI程序
add_executable(myapp WIN32 main.cpp gui.cpp)

# 在macOS上创建一个应用程序包
add_executable(myapp MACOSX_BUNDLE main.cpp app.cpp)

# 将目标排除在默认构建之外
add_executable(extraTool EXCLUDE_FROM_ALL tool.cpp)
```

需要注意的是,`add_executable`通常与`target_link_libraries`命令一起使用,以链接所需的库文件。

总的来说,`add_executable`命令是创建可执行目标的关键命令,它允许你指定可执行文件的名称、目标平台以及构建所需的源文件。

### set  命令与  CMAKE_CXX_STANDARD，CMAKE_CXX_STANDARD_REQUIRED
set 命令是用来给 cmake 中变量赋值的。
基本语法如下:
```cmake
set(<variable> <value>... [PARENT_SCOPE])

# 设置变量
set(MY_VARIABLE "Hello World")
# 设置列表变量
set(MY_LIST_VAR a b c d)
# 追加值到列表变量
set(MY_LIST_VAR ${MY_LIST_VAR} e f)
# 设置缓存变量
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall" CACHE STRING "C++ compiler flags" FORCE)
```
- `<variable>`是要设置的变量名。
- `<value>...`是要赋给变量的值,可以是一个或多个值。如果有多个值,它们将被连接成一个字符串。
- `PARENT_SCOPE`是一个可选的参数,如果指定了它,那么该变量的值将被设置在父作用域中,而不是当前作用域。

**CMAKE_CXX_STANDARD** 是cmake 内置变量，用来指定 C++ 标准的版本。可以将其设置为以下值之一:

- 98 (C++98标准)
- 11 (C++11标准)
- 14 (C++14标准)
- 17 (C++17标准)
- 20 (C++20标准)
- 23 (C++23标准,从CMake 3.20开始支持)

如果没有显式设置该变量,CMake会根据编译器的默认行为来选择C++标准版本。

**CMAKE_CXX_STANDARD_REQUIRED** 
这个变量是一个bool值,用于指定编译器是否必须支持由`CMAKE_CXX_STANDARD`指定的版本。如果设置为`ON`,那么如果编译器不支持该版本,CMake将终止配置阶段并报错。如果设置为`OFF`,CMake将尝试使用可用的最新C++标准版本,而不会报错。
默认情况下,`CMAKE_CXX_STANDARD_REQUIRED`为`OFF`。

### configure_file
`configure_file` 是CMake中一个非常有用的命令,它允许你将一个输入文件转换为另一个输出文件,在转换过程中可以替换文件中的变量值。

该命令的基本语法如下:

```cmake
configure_file(<input> <output>                
				[COPYONLY] [ESCAPE_QUOTES] [@ONLY]               
				[NEWLINE_STYLE [UNIX|DOS|WIN32|LF|CRLF] ])
```

- `<input>` 是要被配置的输入文件
- `<output>` 是转换后的输出文件
- `COPYONLY` 是一个可选的标志,指示只复制文件而不替换其中的变量
- `ESCAPE_QUOTES` 是一个可选的标志,指示是否对变量值中的引号进行转义
- `@ONLY` 是一个可选的标志,指示只替换以`@VAR@`形式的变量值,忽略其他形式
- `NEWLINE_STYLE` 是一个可选的参数,用于指定输出文件的换行符风格

在输入文件中,可以使用`@VAR@`或`${VAR}`这样的语法来引用CMake变量。`configure_file`命令会在转换时将这些变量替换为其实际值。

这个命令的典型用途包括:
1. **生成配置头文件**:通常会将一些编译时的选项、路径等信息写入头文件,供源代码使用。
2. **生成配置文件**:将一些运行时配置信息写入文本文件,供程序在运行时读取。
3. **生成平台相关的文件**:根据不同的目标平台生成不同的资源文件、脚本等。
4. **打印模板文件**:将模板文件中的变量替换后输出最终内容。

`configure_file`命令生成的输出文件的路径取决于输出文件路径的指定方式:
1. **使用相对路径指定输出文件**
如果在`configure_file`命令中使用了相对路径来指定输出文件,那么生成的文件将位于相对于当前`CMAKE_BINARY_DIR`(构建目录)的路径下。

例如:
```cmake
configure_file(config.h.in config.h)
```
这将在`CMAKE_BINARY_DIR`目录下生成`config.h`文件。

2. **使用绝对路径指定输出文件**
如果使用绝对路径指定输出文件,生成的文件将位于该绝对路径所指定的位置,与`CMAKE_BINARY_DIR`无关。

例如:
```cmake
configure_file(config.h.in /usr/local/include/myproj/config.h)
```
这将在`/usr/local/include/myproj/`目录下生成`config.h`文件。

3. **在BINARY_DIR子目录下生成文件**
CMake还提供了一些变量,如`CMAKE_CURRENT_BINARY_DIR`来指定输出文件路径。
```cmake
configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/config.h)
```
这将在`CMAKE_CURRENT_BINARY_DIR/include`目录下生成`config.h`文件。

通常情况下,建议使用相对路径或`CMAKE_CURRENT_BINARY_DIR`指定输出文件路径,这样可以确保生成的文件位于构建目录的子目录中,方便管理和清理。 使用绝对路径需要格外小心,以免将生成的文件直接写入系统目录。
总之,`configure_file`输出文件的具体路径由第二个参数(输出文件路径)决定,开发者需要根据项目需求明智地选择合适的路径。

### target_include_directories
`target_include_directories`是CMake中一个非常重要的命令,它用于为特定的目标添加包含目录(include directories)。包含目录指的是编译器在编译源代码时查找头文件的路径。

该命令的基本语法如下:
```cmake
target_include_directories(<target>
                           [SYSTEM] [BEFORE]
                           <INTERFACE|PUBLIC|PRIVATE> [items1...]
                           [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
```
- `<target>` 是必需参数,指定要为哪个目标添加包含目录。
- `[SYSTEM]` 是一个可选参数,用于将指定的目录标记为系统包含目录,以避免特定于该目录的编译器警告。
- `[BEFORE]` 是一个可选参数,用于在当前目标包含目录的前面添加新的目录。
- `<INTERFACE|PUBLIC|PRIVATE>` 是必需参数,指定了作用域:
  - `INTERFACE`: 添加的目录将作为接口包含目录,对于依赖于该目标的其他目标可见。
  - `PUBLIC`: 添加的目录是当前目标和依赖于当前目标的其他目标的公共包含目录。
  - `PRIVATE`: 添加的目录只用于当前目标的源文件编译。
- `[items1...]` 是一个或多个包含目录的路径,可以使用相对路径或绝对路径。

通过使用不同的作用域,我们可以很好地控制包含目录的可见性,避免头文件污染和冲突。一般情况下:
- 对于项目自身的头文件,应该使用`PRIVATE`作用域。
- 对于外部库的头文件,应该使用`INTERFACE`作用域,方便依赖项目正确包含头文件。
- 对于需要在多个目标之间共享的头文件,可以使用`PUBLIC`作用域。

示例:
```cmake
# 将当前源码目录作为包含目录添加到 myTarget 目标
target_include_directories(myTarget PRIVATE ${PROJECT_SOURCE_DIR})

# 将系统目录 /usr/include 作为接口包含目录添加到 myTarget 目标
target_include_directories(myTarget INTERFACE SYSTEM /usr/include)

# 将外部库头文件目录作为接口包含目录添加到 myTarget 目标
target_include_directories(myTarget INTERFACE 
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/include>
                           $<INSTALL_INTERFACE:include>)
```

正确使用`target_include_directories`命令对于避免头文件相关的编译错误、防止头文件被无意包含等都是非常重要的。在编写CMake脚本时,请务必合理设置包含目录,特别是对于外部库和接口包含目录的处理。

### cmake 内置路径变量
cmake 中有一些常用的内置项目路径变量
#### PROJECT_BINARY_DIR
`PROJECT_BINARY_DIR`是CMake中的一个内置变量,表示构建目录的顶层路径。

具体来说,`PROJECT_BINARY_DIR`变量指向的是在运行CMake时通过命令行参数`-B`或内部选项`--build-dir`指定的构建目录,如果没有显式指定,则使用当前目录。

比如,如果我们在`/path/to/src`目录下运行:
```
cmake -B /path/to/build
```

那么`PROJECT_BINARY_DIR`的值就是`/path/to/build`。

这个变量在CMake脚本中有几个常见的用途:

1. **指定生成文件的输出位置**
通常一些由`configure_file`生成的文件会放置在`PROJECT_BINARY_DIR`或其子目录下,方便与源码区分。
例如:
```cmake
configure_file(config.in ${PROJECT_BINARY_DIR}/config.out)
```

2. **指定中间文件和目标文件的输出位置**
有些CMake命令需要指定输出路径,可以使用`PROJECT_BINARY_DIR`作为基准路径。

3. **定位外部项目和依赖**
外部项目或者依赖库的路径有时候需要相对于`PROJECT_BINARY_DIR`来指定。

4. **CMake调试和测试**
在编写和调试CMake脚本时,知道`PROJECT_BINARY_DIR`的值对于检查生成的中间文件很有帮助。

总的来说,`PROJECT_BINARY_DIR`变量提供了构建目录的位置信息,是CMake脚本中非常重要的一个内置变量,可以帮助我们正确定位构建产物和依赖项的位置。在编写CMakeLists.txt时合理使用它有助于保持项目组织结构的清晰和可维护性。

####  CMAKE_SOURCE_DIR
表示项目根源码目录的完整路径。这个变量在运行CMake时就被设置好了,对于绝大多数项目来说,它的值就是包含顶层`CMakeLists.txt`文件的目录。

#### CMAKE_CURRENT_SOURCE_DIR
表示当前正在处理的`CMakeLists.txt`所在的目录。在递归遍历子目录时,该变量会随之改变,指向当前子目录。

#### CMAKE_CURRENT_BINARY_DIR
表示当前正在处理的`CMakeLists.txt`生成的目标文件和其他构建产物的存放路径。它是`PROJECT_BINARY_DIR`的一个子目录。

#### CMAKE_CURRENT_LIST_DIR
表示当前正在处理的`CMakeLists.txt`文件所在的完整路径。它与`CMAKE_CURRENT_SOURCE_DIR`的区别在于,后者获取的是源码目录,而`CMAKE_CURRENT_LIST_DIR`直接获取`CMakeLists.txt`所在目录。

#### PROJECT_SOURCE_DIR
与`CMAKE_SOURCE_DIR`类似,表示顶层源码目录,但针对的是通过`project()`命令定义的项目。

#### PROJECT_BINARY_DIR
与`CMAKE_BINARY_DIR`类似,表示顶层构建目录,但针对的是通过`project()`命令定义的项目。
