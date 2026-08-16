---
title: 【学习 cmake step5】 安装与测试
date: 2026-08-16T05:54:46.664Z
tags: [C++, cmake]
categories: cpp
---

目标： 安装, 测试
1. 将 MathFunctions 库安装到 `lib` 目录下，头文件安装到 `include` 目录下
2. 将 Tutorial 可执行文件安装到 `bin` 目录下
3. 为 Tutorial 添加单元测试， 并且使用 ctest 来执行测试
## 源码实现
MathFunctions/CMakeLists.txt
```cmake
# TODO1 设置要安装的库 "MathFunctions" 和 "SqrtLibrary", 设置安装目录为 lib
set(installable_libs MathFunctions tutorial_compiler_flags)
if(TARGET SqrtLibrary)
	list(APPEND install_libs SqrtLibrary)
endif()
install(TARGETS ${installable_libs} DESTINATION lib)

# TODO2 设置头文件的安装目录为 include
install(FILES MathFunctions.h DESTINATION include)
```

CMakeLists.txt
```cmake
# TODO3 安装 Tutorial 到 bin 目录，安装头文件到 include 目录
install(TARGETS Tutorial DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h"
	DESTINATION include
)


# TODO 5 开启测试
enable_testing()

# TODO 6 添加一个冒烟测试， 测试 Tutorial 执行文件是否能正常启动
add_test(NAME Runs COMMAND Tutorial 25)

# TODO 7 测试程序是否显示使用说明
add_test(NAME Usage COMMAND Tutorial)
set_tests_properties(Usage
  PROPERTIES PASS_REGULAR_EXPRESSION "Usage:.*number"
  )

# TODO 8 验证程序输出结果是否正确
add_test(NAME StandardUse COMMAND Tutorial 4)
set_tests_properties(StandardUse
  PROPERTIES PASS_REGULAR_EXPRESSION "4 is 2"
  )

# TODO 9 添加更多测试
function(do_test target arg result)
  add_test(NAME Comp${arg} COMMAND ${target} ${arg})
  set_tests_properties(Comp${arg}
    PROPERTIES PASS_REGULAR_EXPRESSION ${result}
    )
endfunction()

# do a bunch of result based tests
do_test(Tutorial 4 "4 is 2")
do_test(Tutorial 9 "9 is 3")
do_test(Tutorial 5 "5 is 2.236")
do_test(Tutorial 7 "7 is 2.645")
do_test(Tutorial 25 "25 is 5")
do_test(Tutorial -25 "-25 is (-nan|nan|0)")
do_test(Tutorial 0.0001 "0.0001 is 0.01")
```

在安装过程中，可以通过 --prefix 参数来指定安装根目录
```bash
cmake --install . --prefix "/home/user_name/install_dir"
```

## install
 `install` 命令用于定义项目的安装规则。它指定了在执行 `make install` 或等效命令时，哪些文件应该被复制到系统中的哪些位置。这个命令非常灵活，可以用于安装各种类型的文件，包括可执行文件、库文件、头文件、配置文件等。

`install` 命令的基本语法如下：
```cmake
install(TARGETS <target1> [<target2> ...]
        [[ARCHIVE|LIBRARY|RUNTIME|OBJECTS|FRAMEWORK|BUNDLE|
          PRIVATE_HEADER|PUBLIC_HEADER|RESOURCE]
         [DESTINATION <dir>]
         [PERMISSIONS permissions...]
         [CONFIGURATIONS [Debug|Release|...]]
         [COMPONENT <component>]
         [NAMELINK_COMPONENT <component>]
         [OPTIONAL] [EXCLUDE_FROM_ALL]
         [NAMELINK_ONLY|NAMELINK_SKIP]
        ] [...])
```

下面是一些常见的用法：

1. **安装可执行文件**
```cmake
install(TARGETS myapp DESTINATION bin)
```
这会将 `myapp` 可执行文件安装到 `${CMAKE_INSTALL_PREFIX}/bin` 目录。

2. **安装库文件**
```cmake
install(TARGETS mylib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
```
这会将动态库安装到 `lib` 目录，静态库安装到 `lib` 目录。

3. **安装头文件**
```cmake
install(FILES myheader.h DESTINATION include/myproject)
```
这会将 `myheader.h` 安装到 `include/myproject` 目录。

4. **安装目录**
```cmake
install(DIRECTORY include/ DESTINATION include/myproject)
```
这会将 `include` 目录下的所有文件安装到 `include/myproject` 目录。

5. **设置安装权限**
```cmake
install(TARGETS myapp 
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
        DESTINATION bin)
```
这设置了安装文件的权限。

6. **根据构建类型安装**
```cmake
install(TARGETS myapp 
        CONFIGURATIONS Debug
        DESTINATION bin/debug)
```
这只在 Debug 构建类型下安装 `myapp`。

7. **安装脚本**
```cmake
install(SCRIPT "some_script.cmake")
```
这在安装过程中运行一个 CMake 脚本。
使用 `install` 命令时，默认的安装前缀是 `CMAKE_INSTALL_PREFIX`，你可以在配置时通过 `-DCMAKE_INSTALL_PREFIX=/path/to/install` 来更改它。

## enable_testing
enable_testing() 是 CMake 中用于启用测试功能的命令。它的主要作用是:

1. 启用项目的测试支持。
2. 允许使用 add_test() 命令添加测试。
3. 生成 "test" 或 "RUN_TESTS" 目标,可以用来运行所有测试。

使用方法非常简单,只需在 CMakeLists.txt 文件中添加这一行:
```cmake
enable_testing()
```
通常,这个命令会放在顶层的 CMakeLists.txt 文件中,以便为整个项目启用测试。

启用测试后,就可以使用 add_test() 命令添加具体的测试了。例如:
```cmake
add_test(NAME MyTest COMMAND MyTestExecutable)
```

## add_test

add_test 命令用于向 CMake 项目添加测试。它告诉 CMake（更具体地说是 CTest）如何运行一个特定的测试。
### 基本语法

最基本的 add_test 语法如下：
```cmake
add_test(NAME <test-name> COMMAND <command> [<arg>...])
```
- NAME：测试的唯一名称
- COMMAND：运行测试的命令（通常是一个可执行文件）
- arg：可选的命令行参数

```cmake
add_test(NAME MyTest COMMAND MyTestExecutable)
```
这会添加一个名为 "MyTest" 的测试，它会运行 "MyTestExecutable" 可执行文件。

带参数的示例
```cmake
add_test(NAME TestWithArgs COMMAND MyTestExecutable arg1 arg2)
```
这个测试会运行 "MyTestExecutable" 并传递 "arg1" 和 "arg2" 作为命令行参数。

**使用生成器表达式**
CMake 允许使用生成器表达式来更灵活地定义测试：
```cmake
add_test(NAME TestWithConfig COMMAND MyTestExecutable --config $<CONFIG>)
```
这会根据当前的构建配置传递正确的配置参数。

**设置工作目录**
你可以指定测试运行的工作目录：
```cmake
add_test(NAME TestInSpecificDir 
         COMMAND MyTestExecutable
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/testdata)
```

**设置环境变量**
可以为测试设置特定的环境变量：
```cmake
add_test(NAME TestWithEnvVar 
         COMMAND MyTestExecutable)
set_tests_properties(TestWithEnvVar 
    PROPERTIES ENVIRONMENT "MY_VAR=my_value")
```

**创建测试套件**
你可以使用命名约定来创建测试套件：
```cmake
add_test(NAME Suite1.Test1 COMMAND Test1)
add_test(NAME Suite1.Test2 COMMAND Test2)
add_test(NAME Suite2.Test1 COMMAND OtherTest1)
```
这样可以使用 ctest -R "Suite1.*" 来运行 Suite1 中的所有测试。

### 与 set_tests_properties 配合使用
add_test 通常与 set_tests_properties 一起使用来设置额外的测试属性：
```cmake
add_test(NAME LongRunningTest COMMAND MyLongTest)
set_tests_properties(LongRunningTest 
    PROPERTIES 
        TIMEOUT 300
        WILL_FAIL FALSE)
```

### 测试依赖设置
虽然 add_test 本身不直接支持测试之间的依赖关系，但你可以使用 set_tests_properties 的 DEPENDS 属性来实现：
```cmake
add_test(NAME Setup COMMAND SetupEnvironment)
add_test(NAME ActualTest COMMAND RunTest)
add_test(NAME Cleanup COMMAND CleanupEnvironment)

set_tests_properties(ActualTest PROPERTIES DEPENDS Setup)
set_tests_properties(Cleanup PROPERTIES DEPENDS ActualTest)
```

### 使用注意事项
- add_test 命令应该在 enable_testing() 之后使用。
- 测试名称在项目中应该是唯一的。
- 添加的测试可以通过 ctest 命令运行。
add_test 命令是 CMake 测试框架的核心，它提供了灵活的方式来定义和配置测试。结合其他 CMake 命令和 CTest 的功能，你可以创建复杂和强大的测试套件。

## 与 CTest 配合
enable_testing() 命令与 CTest 有着密切的关系。
1.  启用 CTest 支持：
   enable_testing() 实际上是为项目启用 CTest 支持的命令。CTest 是 CMake 的测试工具，用于管理和运行测试。
2. 生成 CTest 配置：
   当你使用 enable_testing() 时，CMake 会生成必要的配置文件，使 CTest 能够识别和运行你的测试。
3. 创建测试目标：
   enable_testing() 会创建一个特殊的 "test" 目标（在某些生成器中称为 "RUN_TESTS"）。这个目标可以用来运行所有的测试。
4. 与 add_test() 配合：
   虽然 enable_testing() 本身不添加任何测试，但它允许你使用 add_test() 命令来定义具体的测试。这些测试随后可以被 CTest 执行。
5. CTest 脚本生成：
   enable_testing() 会导致 CMake 生成 CTestTestfile.cmake 文件，这个文件包含了 CTest 运行测试所需的信息。
6. 允许 "ctest" 命令行工具使用：
   启用测试后，你可以使用 "ctest" 命令行工具来运行测试，查看结果，甚至提交到 CDash 等持续集成平台。
enable_testing() 是连接 CMake 项目和 CTest 测试框架的桥梁。它为使用 CTest 进行测试管理和执行做好了准备工作。

### 具体配置
CMakeLists.txt
```cmake
# 在顶层 CMakeLists.txt 中 
cmake_minimum_required(VERSION 3.10) 
project(MyProject) 

# 启用测试 
enable_testing() 

# 添加子目录 
add_subdirectory(src) 
add_subdirectory(tests)
```

tests/CMakeLists.txt
```cmake
# 在 tests/CMakeLists.txt 中 
# 添加测试代码作为可执行文件
add_executable(MyTest test_main.cpp) 
target_link_libraries(MyTest PRIVATE MyLibrary) 

# 添加测试：调用可执行文件进行测试
add_test(NAME MyFirstTest COMMAND MyTest)
```

### ctest 常用命令
编译项目后，你可以使用以下 CTest 命令：
- `ctest`: 运行所有测试
- `ctest -N`: 列出所有测试但不运行
- `ctest -R "regex"`: 运行名称匹配正则表达式的测试
- `ctest -VV`: 详细模式，显示更多输出

## 集成 Google Test
###  添加 Google test 到项目

```cmake
# 从 github 远程添加
include(FetchContent)
FetchContent_Declare(
	googletest 
	GIT_REPOSITORY https://github.com/google/googletest.git 
	GIT_TAG release-1.11.0
)
FetchContent_MakeAvailable(googletest)

# 从本地添加
add_subdirectory(path/to/googletest)
```

### 配置 CMake
主 CMakeLists.txt 文件
```cmake
cmake_minimum_required(VERSION 3.14) 
project(MyProject) 

# 启用测试 
enable_testing() 
# 添加源代码 
add_subdirectory(src) 
# 添加测试 
add_subdirectory(tests)
```

tests/CMakeLists.txt 
```cmake
# 链接 Google Test 
include(GoogleTest) 
# 创建测试可执行文件 
add_executable(MyTests test_feature1.cpp test_feature2.cpp) 
# 链接 Google Test 和你的项目库 
target_link_libraries(MyTests PRIVATE gtest_main MyProjectLibrary ) 
# 发现测试 
gtest_discover_tests(MyTests)
```

高级配置
```cmake
# 设置测试超时 10ms
set_tests_properties(MyTests PROPERTIES TIMEOUT 10) 
# 添加标签 
set_tests_properties(MyTests PROPERTIES LABELS "unit")
```

自定义测试主函数
gtest 提供了 `gtest_main` 作为默认的测试入口，如果要使用自定义的主函数，可以按照下述步骤实施。

tests/CMakeLists.txt 
```cmake
add_executable(MyTests test_main.cpp test_feature1.cpp test_feature2.cpp) 

target_link_libraries(MyTests PRIVATE gtest MyProjectLibrary)
```

test_main.cpp
```c++
#include <gtest/gtest.h>

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
```


**构建&执行测试**
```bash
# 构建项目
cmake -S . -B build 
cmake --build build

# 执行测试
ctest -j4 # 并行运行 4 个测试
```