---
layout: post
title: 【学习 cmake step2】 创建公共库(library)
date: 2024-01-04 00:00:00 +0800
categories: C++ cmake
typora-root-url: ..
---


目标：学习在项目中如何构建和使用一个公共库(library)。

1. 在项目子文件夹中创建一个公共库，名字叫做 `MathFunctions` 供项目使用。
2. 提供一个编译选项，来选择是使用 `MathFunctions` 还是系统库。 



# 源码实现

[Step 2: Adding a Library](https://cmake.org/cmake/help/latest/guide/tutorial/Adding%20a%20Library.html)

**目标1** 创建 MathFunctions 公共库 
Step2/MathFunctions/CMakeLists.txt

```cmake
# TODO 1: 添加子目录名字叫做 "MathFunctions" 的库 (但是没有指定库的类型)
add_library(MathFunctions MathFuntions.cxx mysqrt.cxx)
```

Step2/CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.10)
project(Tutorial VERSION 1.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
configure_file(Tutorial.h.in TutorialConfig.h)

# TODO 2: 添加子目录"MathFunctions"作为依赖库
add_subdirectory(MathFunctions)

add_executable(Tutorial tutorial.cxx)

# TODO 3: 将公共库添加到编译链接中
target_link_libraries(Tutorial PUBLIC MathFunctions)
# TODO 4: 添加公共库的头文件到inlucde中
target_include_directories(
	Tutorial PUBLIC
	"${PROJECT_BINARY_DIR}"
	"${PROJECT_SOURCE_DIR}/MathFunctions"
)
```

**目标2** 选择编译，使用自定义的 `sqrt` 函数还是 `std::sqrt`
通过添加编译参数 `USE_MYMATH` 并且将 `sqrt` 函数的实现代码 `mysqrt.cxx` 单独编译成一个公共库 `SqrtLibrary` 来实现。
Step2/MathFunctions/CMakeLists.txt
```cmake
# TODO 1: 添加子目录名字叫做 "MathFunctions" 的库 (但是没有指定库的类型)
# add_library(MathFunctions MathFuntions.cxx mysqrt.cxx)

# TODO 14: 将 mysqrt.cxx 从 MathFunctions 中移除
# 注意这里定义 MathFunctions 为动态库
add_library(MathFunctions SHARED MathFuntions.cxx)

# TODO 7: 创建控制变量 USE_MYMATH 默认值为 ON
option(USE_MYMATH "Use tutorial provided math implementation" ON)

# TODO 8: 设置编译变量 'USE_MYMATH' 来控制代码编译内容
if (USE_MYMATH)
	target_compile_definitions(MathFunctions PRIVATE "USE_MYMATH")
endif ()

if (USE_MYMATH)
	# TODO 12: 将自己实现的 mysqrt.cxx 定义为公共库 SqrtLibrary
	# 定义为静态库
	add_library(SqrtLibrary STATIC mysqrt.cxx)
	# 或者定义为动态库
	# add_library(SqrtLibrary SHARED mysqrt.cxx)

	# TODO 13： 添加编译链接
	target_link_libraries(MathFunctions PRIVATE SqrtLibrary)
endif ()
```

Step2/MathFunctions/MathFunctions.cxx
```c++
#include "MathFunctions.h"
// TODO 11: 添加标准库 cmath
#include <cmath>

// TODO 10: 根据编译参数引入 mysqrt.h
#ifdef USE_MYMATH
#include "mysqrt.h"
#endif

namespace mathfunctions {
double sqrt(double x)
{
// TODO 9: 根据编译参数 USE_MYMATH 来选择具体调用的 sqrt 函数
#ifdef USE_MYMATH
return detail::mysqrt(x);
#else
return std::sqrt(x);
#endif

}
}

```

# 命令讲解

## add_library

`add_library`命令在CMake中用于定义一个库目标。它有以下几种使用方式:

1. **静态库**
```cmake
add_library(lib_name [STATIC] source1.cpp source2.cpp ...)
```
这将使用`source1.cpp`、`source2.cpp`等源文件创建一个名为`lib_name`的静态库。

2. **共享库(动态库)**
```cmake 
add_library(lib_name [SHARED] source1.cpp source2.cpp ...)
```
这将使用`source1.cpp`、`source2.cpp`等源文件创建一个名为`lib_name`的共享库。

3. **对象库**
```cmake
add_library(lib_name OBJECT source1.cpp source2.cpp ...)
```
这将编译`source1.cpp`、`source2.cpp`等源文件,但不会创建库文件,而是创建一个对象库,以便于后续链接到其他目标。

4. **导入库**
```cmake
add_library(lib_name [STATIC | SHARED | OBJECT] IMPORTED [IMPORTED_LOCATION])
```
这种方式用于导入一个预先构建好的库,而非使用源代码构建库。需要指定库的类型(`STATIC`、`SHARED`或`OBJECT`)以及它的位置。
在定义库目标之后,你可以使用`target_link_libraries`命令将其链接到可执行目标或其他库目标。同时,也可以使用`set_target_properties`等命令设置库目标的属性。

在使用 `add_library` 来创建库时，如果不指定库的类型(STATIC, SHARED)， cmake 可能根据不同操作系统平台，默认指定库的类型。

## add_subdirectory
`add_subdirectory`命令用于将另一个独立的CMakeLists.txt文件包含到当前的CMake构建中。它具有以下几个作用:

1. **包含子目录**
最常见的用法是将位于子目录中的CMakeLists.txt文件包含进来。这样子目录中定义的目标(如库、可执行文件等)也会被构建。
```cmake
add_subdirectory(subdirName)
```

2. **包含外部项目**
如果有一个外部项目提供了自己的CMakeLists.txt文件,我们也可以使用`add_subdirectory`命令将它包含进当前项目构建中。
```cmake
add_subdirectory(/path/to/external/project)
```

3. **指定二进制目录**
`add_subdirectory`还可以指定一个二进制目录,用于存放编译生成的中间文件和目标文件。
```cmake
add_subdirectory(subdirName ${PROJECT_BINARY_DIR}/generated)
```

4. **排除构建某些组件**
可以通过设置CMake变量来控制是否包含某些子目录。
```cmake
option(WITH_SUBDIR "Build subdirectory?" ON)
if(WITH_SUBDIR)
    add_subdirectory(subdirName)
endif()
```

5. **控制子目录构建行为**
`add_subdirectory`还可以传递额外的参数,这些参数会被传递给子目录的顶层CMakeLists.txt文件。子目录可以使用这些参数来控制其自身的构建行为。
总之,`add_subdirectory`命令提供了一种方便的机制,将其他CMakeLists.txt文件集成到当前的构建系统中。它支持跨目录组织项目,以及复用外部项目。

如果要同时指定多个包含 CMakeLists.txt 文件的子目录。
方案一： 分别使用多个add_subdirectory命令
```cmake
add_subdirectory(subdir1) 
add_subdirectory(subdir2) 
add_subdirectory(subdir3)
```
使用这种方式,每个子目录会被分别包含,CMake会按照命令出现的顺序逐个处理子目录。这在处理有依赖关系的子目录时很有用。

方案二： 使用单个add_subdirectory命令并传递多个子目录
```cmake
add_subdirectory(subdir1 subdir2 subdir3)
```
这种方式更加简洁,一行就可以包含多个子目录。CMake会按照参数列表的顺序依次处理每个子目录。
不过需要注意,第二种方式中,所有子目录会被视为在同一级别,CMake不会去分析它们之间是否存在依赖关系。如果存在依赖关系,需要确保参数列表中的顺序是正确的。
另外,无论采用哪种方式,被包含的子目录都需要包含一个CMakeLists.txt文件。如果某个子目录不包含该文件,CMake会直接忽略它。

## target_link_libraries
`target_link_libraries`命令是CMake中用于为目标(executable或library)链接所需库的关键命令。它的主要作用如下:

1. **链接库目标**
可以链接之前使用`add_library`定义的库目标。
```cmake
add_library(mylib src1.cpp)
add_executable(myapp main.cpp)
target_link_libraries(myapp mylib)
```

2. **链接系统库**
可以链接系统库,如pthread、m等。
```cmake
target_link_libraries(mytarget pthread m)
```

3. **链接第三方库**
可以链接第三方库,如boost、gtest等。
```cmake
target_link_libraries(mytarget /path/to/third_party_lib.so)
```

4. **链接导入库**
可以链接使用`add_library(... IMPORTED)`导入的预构建库。

5. **控制链接选项**
可以传递链接选项,如链接时间优化选项(-flto)、静态/动态链接选项(-static)等。
```cmake
target_link_libraries(mytarget -flto -static)
```

6. **传递链接依赖**
如果A链接了B和C,当D链接A时,B和C也会被传递性地链接进来。
`target_link_libraries`是非常灵活和强大的,几乎所有链接目标库的情况都需要使用它。正确使用它可以确保目标正确构建和链接所需的所有依赖库。
此外,`target_link_libraries`还支持传递链接接口,即被链接目标所暴露的包含目录、链接选项等,这使得构建更加模块化。
总之,`target_link_libraries`是CMake中管理目标库依赖关系的核心命令,熟练掌握它对于跨平台C++构建至关重要。


## option
`option`命令在CMake中用于定义一个选项变量,这个变量可以在运行cmake时或者通过其他界面(如cmake-gui)进行设置。它的语法如下:
```cmake
option(<option_variable> "help_string" [value])
```
- `<option_variable>`是将要定义的选项变量名。
- `"help_string"`是一个描述该选项作用的帮助字符串。
- `[value]`是可选的,用于指定默认值。如果省略,默认为OFF。

例如:
```cmake
option(WITH_OPENGL "Build with OpenGL support" ON)
```
这条命令定义了一个名为`WITH_OPENGL`的选项变量,描述为"Build with OpenGL support",默认值为ON。

在定义选项后,可以在配置CMake时通过命令行或其他界面设置该选项的值。例如:
```
cmake -DWITH_OPENGL=OFF ..
```
或者通过cmake-gui等图形界面进行设置。

定义的选项变量可以在CMake脚本中使用,通常用于条件编译。比如:
```cmake
if(WITH_OPENGL)
    add_definitions(-DUSE_OPENGL)
    ...
else()
    ...
endif()
```
`option`命令为CMake构建系统提供了灵活性,允许用户根据需要启用或禁用某些特性。它使得一次配置就可以构建出不同的版本。开发人员也可以通过设置默认值,方便地控制构建行为。

`option` 与 `set` 命令对比



| 特性         | option 命令                           | set 命令                                          |
| ------------ | ------------------------------------- | ------------------------------------------------- |
| 定义方式     | 定义布尔选项变量,只能取 ON 或 OFF 值  | 可定义任意类型变量,包括布尔、字符串、列表等       |
| 用户可操作性 | 可在运行 CMake 时通过命令行或界面设置 | 无法在运行时直接修改,需在 CMakeLists.txt 中硬编码 |
| 默认值       | 如果未指定,默认为 OFF                 | 必须显式指定值                                    |
| 帮助信息     | 可添加描述选项用途的帮助字符串        | 无法添加帮助信息                                  |
| 主要目的     | 为用户提供配置选项,控制构建行为       | 通用变量定义,用途广泛                             |

两者虽然都可以定义变量,但 `option` 命令更侧重于为用户提供可配置的选项,而 `set` 命令则更加通用和灵活。合理使用两者有助于提高 CMake 构建系统的可用性和可维护性。



## target_compile_definitions

`target_compile_definitions` 是 CMake 中用于为特定目标(target)添加编译定义的命令,通常用于条件编译。它的语法如下:
```cmake
target_compile_definitions(<target>
                           <INTERFACE|PUBLIC|PRIVATE>
                           [items1...]
                           [...]
                           )
```
- `<target>` 是要添加编译定义的目标,可以是由 `add_executable`、`add_library` 等命令创建的可执行目标或库目标。
- `<INTERFACE|PUBLIC|PRIVATE>` 用于指定编译定义的作用域:
  - `INTERFACE`: 将定义传播给依赖于该目标的其他目标。
  - `PUBLIC`: 将定义传播给依赖于该目标的其他目标,并将其合并到依赖于上一个目标的目标中。
  - `PRIVATE`: 定义仅用于该目标本身,不会传播到其他目标。
- `[items...]` 是一个或多个编译定义的列表,每个定义可以是 `KEY=VALUE` 形式或只指定 `KEY`。

例如:
```cmake
add_library(mylib src1.cpp src2.cpp)
target_compile_definitions(mylib
  PRIVATE 
    DEBUG_LEVEL=2   # 为mylib指定DEBUG_LEVEL=2
    USE_CUSTOM_ALLOC # 为mylib启用USE_CUSTOM_ALLOC
  PUBLIC   
    MYLIB_EXPORTS   # 将MYLIB_EXPORTS传播到依赖于mylib的其他目标
)
```

添加到目标上的编译定义会被传递给编译器,在源文件编译时生效。它们主要用于:
1. 条件编译代码
2. 定义构建配置相关的宏
3. 控制库或程序的行为
通过 `target_compile_definitions` 可以清晰地管理每个目标所需的编译定义,使构建系统更加模块化和可维护。相比在源代码中使用 `#define`、`-D` 编译器标志等方式,它更加集中和便于控制。

对比 `target_compile_definitions` 和 `add_definitions`

| 对比项  | target_compile_definitions     | add_definitions  |
| ---- | ------------------------------ | ---------------- |
| 作用范围 | 只针对指定的目标(target)               | 对所有目标都生效,全局范围    |
| 传递性  | 可通过`INTERFACE`、`PUBLIC`传递给依赖目标 | 不会传递给依赖目标        |
| 可重用性 | 模块化,更利于代码重用                    | 全局性,不利于代码重用      |
| 生命周期 | 与目标生命周期相同,更易管理                 | 在整个构建过程中都有效,难以管理 |
| 作用时间 | 立即为指定目标添加编译定义                  | 直到遇到第一个目标定义时才生效  |



总的来说,`target_compile_definitions`更加现代、灵活、模块化,符合CMake的目标导向设计理念,建议优先使用。而`add_definitions`是一种传统的、全局的方式,在某些简单场景下也可以使用,但不够灵活。

在编写CMakeLists.txt时,尽量使用`target_compile_definitions`以提高代码的可重用性和可维护性。



## 与位置无关的代码 (-fPIC)

在目标2 的代码中如果将 `MathFunctions` 定义为动态库，而`SqrtLibrary` 定义为静态库，那么在编译过程中会报错：
<code>/usr/bin/ld: libSqrtLibrary.a(mysqrt.cxx.o): relocation R_X86_64_PC32 against symbol `_ZSt4cout@@GLIBCXX_3.4` can not be used when making a shared object; recompile with -fPIC</code>

**先解释一下什么是 PIC**
PIC 的全称是 **Position-Independent Code** 与位置无关的代码, 它是指这段代码在加载和执行时,可以被放置在内存的任何位置而不会产生错误。
这个概念的背景是,在现代操作系统中,进程的虚拟内存地址空间分为多个区域,不同类型的代码和数据被加载到不同的区域。而这些区域的起始地址并不是固定的,每次加载时都可能不同。如果代码中存在一些对内存地址做了绝对假设的情况,就可能导致程序在某些特定内存布局时无法正常运行。为了避免这种情况,需要生成"与位置无关的代码"(PIC)。具体来说,PIC代码在访问数据时,使用相对寻址而不是绝对地址,从而避免了对特定内存地址的依赖。编译器在生成PIC代码时,会插入一些额外的指令来计算数据的相对位置。
生成PIC代码是必要的,主要有以下几种情况:
1. **生成共享库(动态库)时**,共享库被加载到内存后其地址是不确定的,必须使用PIC。
2. **使用动态加载技术**时,动态加载的代码也必须是PIC。
3. **内核模块编程**时,内核模块加载位置也是不确定的,需要PIC。
4. 某些**特殊的嵌入式系统**对代码的内存布局有特殊要求,也需要PIC。
可以通过给编译器传递`-fPIC`选项来生成PIC代码。生成PIC代码会带来少量的代码大小和性能开销,但在现代CPU上这个开销通常可以忽略。

**那为什么在编译项目的时候会爆这个错误呢？**
因为我们定义 `MathFunctions` 为动态库，而 `SqrtLibrary` 为静态库；如果**最终的可执行程序**或者**静态库**需要链接**动态库**, 则其中的所有代码都需要是位置无关代码(PIC)。这是因为动态库在加载时,其位置是不确定的,如果可执行程序中有非PIC代码,就会导致动态库无法正常加载和运行。

另外，从glibc 2.23版本开始,链接器开始强制要求静态库必须是PIC格式的,否则会拒绝链接。之前的glibc版本也建议使用PIC静态库,但不是强制的。这一变化主要是为了执行保护和加固功能。因此,如果要在Linux系统上构建一个可执行程序,并且该程序需要链接动态库,或者要链接glibc 2.23+版本,那么所有被链接进来的静态库都必须是PIC格式的。如果之前的静态库是以默认非PIC方式编译的,就需要加上`-fPIC`选项重新编译,生成PIC格式的静态库。这一要求不仅来自于动态库加载的需求,也是glibc对静态库格式的限制要求。

**解决方法**
1. 将 `MathFunctions` 和 `SqrtLibrary`  都定义为动态库
2. 为 cmake 添加上参数  `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`
3.  通过 `set_target_properties(SqrtLibrary PROPERTIES POSITION_INDEPENDENT_CODE)`  单独将 SqrtLibrary  设置为支持 PIC。 

