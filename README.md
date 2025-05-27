## 相关配置

### opencv

先源码编译opencv-4.11.0，其中编译注意事项参考如下：

[Qt配置OpenCV教程，亲测已试过（详细版）_qt opencv-CSDN博客](https://blog.csdn.net/weixin_43763292/article/details/112975207)

注意，对于vs的编译器MSVC，其依赖的库为lib和dll，而对于MinGW编译器（通用编译器）其依赖的是.a

报错：

[学习OpenCV3：MinGW编译OpenCV到vs_version.rc.obj处出错_mingw 编译opencv gcc: error: long: no such file or d-CSDN博客](https://blog.csdn.net/qq_34801642/article/details/105583164)

[error: ‘std::_hypot‘ has not been declared using std::hypot；_error: ‘::hypot’ has not been declared using ::hyp-CSDN博客](https://blog.csdn.net/Vertira/article/details/132633070)

qmake配置

```c
INCLUDEPATH += D:\opencv\opencv-build\install\include
LIBS += D:\opencv\opencv-build\lib\libopencv_*.a
```

### pt模型转换为onnx模型

可使用python代码或看官方文档：

```Python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Export the model to ONNX format
model.export(format="onnx", opset=12)  # creates 'yolo11n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("yolo11n.onnx")
```

### 另外说明，这只是一个简单的demo，另提供一个demo运行在华为昇腾AI上的推理代码在ascend_yolo文件夹中

其中需注意onnx转为为om（华为昇腾AI）需要的模型指令代码（示例如下，可以去官网详细看并修改）：

```C++
atc --model=yolo11n.onnx --framework=5 --output=yolo11n_rgb --input_shape="images:1,3,640,640" --soc_version=Ascend310B4 --insert_op_conf=aipp_rgb.cfg
```

```C++
 
aipp_rgb.cfg文件内容如下 
    
aipp_op{
    aipp_mode:static
    input_format : YUV420SP_U8
    src_image_size_w : 640
    src_image_size_h : 640

    csc_switch : true
    rbuv_swap_switch : false
    matrix_r0c0 : 256
    matrix_r0c1 : 0
    matrix_r0c2 : 359
    matrix_r1c0 : 256
    matrix_r1c1 : -88
    matrix_r1c2 : -183
    matrix_r2c0 : 256
    matrix_r2c1 : 454
    matrix_r2c2 : 0
    input_bias_0 : 0
    input_bias_1 : 128
    input_bias_2 : 128

    crop: true
    load_start_pos_h : 0
    load_start_pos_w : 0
    crop_size_w : 640
    crop_size_h : 640

    min_chn_0 : 0
    min_chn_1 : 0
    min_chn_2 : 0
    var_reci_chn_0: 0.0039215686274509803921568627451
    var_reci_chn_1: 0.0039215686274509803921568627451
    var_reci_chn_2: 0.0039215686274509803921568627451
}
```

