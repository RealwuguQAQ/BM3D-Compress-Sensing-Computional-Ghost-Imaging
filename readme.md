#基于BM3D的压缩感知鬼成像算法

这是一个利用BM3D实现的压缩感知鬼成像算法，本代码复现**Vladimir Katkovnik**和**Jaakko Astola**的matlab工作，并且将其转换为Python版本的实现。
本代码对源代码中的闭源BM3D库函数利用pip包进行了替换，可以在代码基础上进行进一步修改研究

[参考文献](https://opg.optica.org/josaa/abstract.cfm?URI=josaa-29-8-1556)

[参考源代码](https://opg.optica.org/josaa/abstract.cfm?URI=josaa-29-8-1556)

##How To Run
首先你的环境必须包含以下包

|  package  | version  |
|  ----  | ----  |
| bm3d  | 4.0.3 |
| numpy  | 2.0.2 |
| scipy  | 1.13.1 |

在完成环境安装并且保证无报错后，运行
```
python3 demo_Compresdsive_Sensing_Ghost_Imaging.py
```

##How to connect me
我的邮箱:wuguqaq@hotmail.com

鬼成像相关研究大部分都是闭源，本人也是一位在该领域刚开始研究的新人，在研究过程中难免遇到卡壳，欢迎相关研究者能够与我联系交流，共同进步！
