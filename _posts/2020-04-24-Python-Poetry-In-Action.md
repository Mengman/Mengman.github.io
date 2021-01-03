---
layout: post
title: "Python 依赖管理软件 Poetry 上手"
date: 2020-04-20 17:26:21 +0800
categories: designpattern softwareengineering
---

## Python 项目管理 Poetry 上手

Python 项目依赖管理工具有 virtualenv pipenv anaconda，来帮助项目做到项目之间的环境隔离，依赖管理等。今天介绍的 Poetry 是它更像是 Python 项目 CLI 工具，能够进行项目模板创建，依赖管理，项目构建与发布等。Poetry v0.1.0 版本发布2018-02，到现在两年的时间，目前社区比较活跃。

### Poetry 配置

#### 1  安装

OSX / Linux / bashonwindows 使用脚本自动安装

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

在类 *nix 的系统中 Poetry 会默认安装在 <code>$HOME/.poetry/bin </code> 中， 在window系统下默认安装目录为 <code>%USERPROFILE%\.poetry\bin</code>。

如果要修改安装目录，需要在运行安装脚本时通过环境变量 <code>POETRY_HOME</code> 来指定。 使用参数 <code>--version</code> 或者环境变量 <code>POETRY_VERSION</code> 可以指定安装版本。

```bash
POETRY_HOME=/etc/poetry python get-poetry.py --version 1.0.5
```

#### 2. 安装检查

安装成功后检查 poetry 版本

```bash
poetry --version
```

如果无法找到  <code>poetry</code> 命令， 那么记得把安装目录加到 $PATH 中。

```bash
export $PATH=$HOME/.poetry/bin:$PATH
```

#### 3.更新&卸载 Poetry

更新 Poetry 

```bash
# 更新到最新的稳定版本
poetry self update

# 更新到预览版本
poetry self udpate --preview

# 更新到指定版本
poetry self update 1.0.5
```

卸载 Poetry

```bash
python get-poetry.py --uninstall
```



### 基本用法



