---
layout: post
title: "pkg-config 入门"
date: 2020-07-26 00:41:00 +0800
categories: c++ linux
---

# pkg-config 入门

pkg-config 是类 Unix 操作系统上的一个库查询工具。它通过 .pc 文件来查询当前系统上已安装类库的信息。

下面是 OpenCV4 的 pc 文件， 文件中包含了OpenCV 库的很多元数据(metadata)

```
# Package Information for pkg-config

prefix=/usr/local/opencv4
exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir_old=${prefix}/include/opencv4/opencv
includedir_new=${prefix}/include/opencv4

Name: OpenCV
Description: Open Source Computer Vision Library
Version: 4.0.1
Libs: -L${exec_prefix}/lib -lopencv_gapi -lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_cvv -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_sfm -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_videostab -lopencv_video -lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core
Libs.private: -ldl -lm -lpthread -lrt
Cflags: -I${includedir_old} -I${includedir_new}
```

pc 文件中的元数据分为两种，一种是格式如 <code>prefix=**</code> 的变量，一种是格式如 <code>Name: OpenCV</code> 关键字，变量是在 pc 文件内部使用，用来帮助定义关键字；而关键字信息会导出到 pkg-config 中。

pkg-config 中定义了一下这些关键字

* Name: 当前库的名字，它不需要全局唯一，通常它和 pc 文件的文件名保持一致。
* Description: 对于库的简短描述
* URL: 当前库的相关信息网络地址
* Version: 安装的版本号
* Requires: 列出当前库所需要的依赖，依赖库的版本信息可以使用 =, >, <, <= 和 >= 来指定。 eg. <code>Requires: OpenCV >= 4.0.0</code> 
* Requires.private: 列出当前库所需要的依赖，但是这些依赖并不会暴露给应用，版本格式与 <code>Requires</code> 相同。
* Confilcts: 这是一个可选关键字，用来指明与当前库冲突的库
* Cflags: 编译当前库需要的 flags 参数。如果依赖的库不支持 pkg-config， 就可以通过此关键字来指定，如果支持 pkg-config，就应该在 <code>Requires</code> 和 <code>Requires.private</code> 中指定。
* Libs: 指定不支持 pkg-config 的链接依赖 flags 参数
* Libs.private: 同 <code>Requires.private</code> 一样，当前关键字的依赖不会暴露给应用



pkg-config 默认会在 <code>/usr/lib/pkgconfig</code> 和 <code>/usr/share/pkgconfig</code> 中去加载 pc 文件，但是有些库的安装地址不在默认路径中，就需要将 pc 文件的路径添加到环境变量 **PKG_CONFIG_PATH** 中。



通过 pkg-config 可以查看安装库的版本

```bash
pkg-config --modversion opencv4 # 如果 pc 文件名与 Name 关键字不一致，这里需要输入的是文件名
4.0.1
```

<code>--libs</code> 可以输出编译依赖当前库的 link flags， 但是不会输出 /usr/lib 下的依赖库。

```bash
pkg-config --libs opencv4
-L/usr/local/opencv4/lib -lopencv_gapi -lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_cvv -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_sfm -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_videostab -lopencv_video -lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core
```

同时 <code>--libs</code> 也不会输出 OpenCV 的依赖，因为依赖 OpenCV 的代码并不直接依赖与OpenCV 的依赖。为了静态链接，可以加上选项 <code>--static</code>

```bash
pkg-config --libs opencv4
-L/usr/local/opencv4/lib -lopencv_gapi -lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_cvv -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_sfm -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_videostab -lopencv_video -lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core -ldl -lm -lpthread -lrt
```

使用 <code>--cflags</code> pkg-config 会输出所有的编译flag， 加不加 <code>--static</code> 都一样。

```bash
pkg-config --cflags opencv4
-I/usr/local/opencv4/include/opencv4/opencv -I/usr/local/opencv4/include/opencv4

pkg-config --cflags --static opencv4
-I/usr/local/opencv4/include/opencv4/opencv -I/usr/local/opencv4/include/opencv4
```

使用 pkg-config 可以更加方便的编译C++代码，比如代码 <code>myapp.cpp</code> 依赖 OpenCV，那么可以使用下面方式来指定编译的flags

```bash
g++ `pkg-config --cflags --libs opencv4` -o myapp myapp.cpp
```

