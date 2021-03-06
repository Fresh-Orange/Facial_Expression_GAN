## 任务简介

这个项目是我参加第一届[BIGO短视频内容智能制作技术挑战赛](http://prcv2019.ml.bigo.sg/)的解决方案。生成的结果获得了人脸生成任务的二等奖。

### 数据集

这一比赛给出的训练集是若干个视频，以及每个视频中每一帧的人脸bbox和人脸关键点坐标（104个关键点）。要求给出一个模型能够以人脸图片和人脸关键点序列作为输入，并生成外观与参考图片一致且人脸表情与输入关键点序列一致的视频序列。任务示例如下

![image-20200410170014316](images/renwu.png)



数据集： https://share.weiyun.com/54e3ZQ0 （数据归BIGO公司所有，如有研究之外的用途请自行联系BIGO）

测试数据将根据生成难度分为`简单`，`适中`，`困难`三个等级。人脸表情生成的难易度则与人脸的角度变化幅度和参考图片中人脸的挂饰的丰富度成正比。即在一段测试序列中，人脸角度变化幅度越大，人脸挂饰越多，测试难度越大。

上面的示例只是最简单的情况，而在实际的样本中

- 人脸可能只占据了画面的很小一部分
- 可能是抬头低头抑或是侧脸
- 存在人脸被部分遮挡甚至不在画面内的情况

### 结果评价

对结果的评价指标有以下四个：

1. 图像质量评价指标PSNR
2. 图像质量评价指标SSIM
3. 人脸身份分类精度
4. 人脸关键点检测精度



## 解决方案

### 预处理

- 裁剪出人脸。因为每帧图像中不止有人脸，还有身体部分和复杂的背景。为了更好的生成人脸，我首先利用人脸的bbox信息将人脸裁剪出来，训练集和测试集都是如此。

  `dataset/face_crop.py`中的`crop_face`函数就是用来将人脸从原视频中裁剪下来的，而其中的`draw_bbox_keypoints`函数则是将人脸关键点的坐标绘制成背景为黑色的关键点图

- 数据增强。在实验过程中发现人脸旋转和转头的时候模型生成的人脸存在重影等问题，因此对人脸进行了旋转的增强方式。这里需要注意的是，由于需要对多张图像进行相同的旋转，所以不能直接使用randomRotate等函数。

  



### 条件GAN生成人脸

GAN的部分是这个项目的主体部分，首先用一张图说明一下模型架构：

![image-20200410221805750](images/moxing.png)

模型方面我做了很多尝试，除了根文件夹中的`main`文件还包括`discarded`文件夹中的，最终我采用的是`main_ResId_finetune_aug.py`这个方案，模型的具体实现如下：

#### 生成器

使用的是stargan中的生成器，输入通道数为6以适应人脸图+关键点图的输入，生成器的训练目标是输出人脸的身份信息和输入人脸相同，但是关键点与输入的关键点相同的人脸。当然这个生成器也存在一定问题，例如存在棋盘效应等等，以后如果再用的话可以选择多尝试其他架构的生成器。



#### 判别器---真假和身份

- 一开始判别器使用的就是stargan中的判别器，或者可以说是patchgan中的判别器
- **真假判别器**：与stargan不同的是stargan中判别器的输出不仅有真假概率，而且还有domain分类概率，然而放在该项目中真假判别器的输出就仅仅是真假概率
- **身份判别器**：为了使得判别器能判别身份，对判别器进行了修改，将输入改成了6通道的输入，分别是源人脸和生成的人脸，输出也是真假概率，具体在这里“真”表示的是两个输入是同一身份，“假”表示的是两个输入是不同身份
- **SNConv2d**：对原始GAN的改进有WGAN，WGAN-GP等等方式，这些方法需要对判别器的梯度做裁剪或者做惩罚，stargan中的方式就是WGAN-GP的梯度惩罚方法。但是*Spectral Normalization for Generative Adversarial Networks*这篇论文使用一种更优雅的方式使得判别器 D 满足利普希茨连续性，限制了函数变化的剧烈程度，从而使模型更稳定。 改用SNConv2d的方法也很简单，首先是原判别器中的Conv2d改成SNConv2d（具体实现查看代码），其次是输出的真假概率加上softplus函数处理，当然原本的梯度惩罚的步骤就删除了。 根据实验可知使用该判别器能对效果改进很多！然而惭愧的是，其原理我并没有很理解。



#### 关键点检测器

- 这个模块输入为一张人脸图像，输出为一张关键点图像。
- 模型的实现其实是直接拿stargan生成器，所以关键点检测器和生成器在模型方面是完全等同的
- 训练时用的是数据集中的人脸和对应的关键点，然后该模型被用于检测生成人脸（假人脸）的关键点，用以保证生成的人脸的关键点和目标关键点能匹配。
-  很重要的一个技巧：**训练该模型时需要对输入的人脸图像加随机噪声，不然该模型会太过拟合以至于对假人脸检测失效**
- 在对抗网络中还有一点需要注意，就是在GAN每轮训练过程中，关键点检测器也要保持更新，否则会被生成器钻空子。也就是说如果关键点检测器是一成不变的，那么生成器就能只能知道怎么“作弊”从而骗过关键点检测器。
- 这一模块的损失函数为生成的关键点图和真实关键点图的L1损失，**这个损失肯定存在缺点，但我目前没有好的改进方向**



#### 背景融合

首先声明这个部分是一个病态的问题，是这个比赛问题定义与实际情况不符的表现。

- **解决什么问题**。这个比赛要求通过一帧人脸图像和目标人脸关键点位置去生成一帧人脸图像。但是要知道一帧图像中不仅仅有人脸部分，还有例如头发，身体，背景等等，这些信息是没法通过目标人脸关键点得到的。例如下面这张：

  <img src="images/ronghe.png" alt="image-20200417113620263" style="zoom:50%;" />

- **第一种解决方法**——平移和缩放整个输入帧，这是很自然的想法，就是将输入的bbox和生成帧的bbox对齐，而对齐的过程中会对输入帧进行平移（例如上图的情况需要想左上方平移）以及缩放（例如上图中原bbox和目标bbox的大小不同需要缩放）。而平移和缩放会引入新的问题，就是可能需要进行边缘填充，这里也有两个方法 ① 用边缘复制法进行了填充，但是也针对不同难度的数据集进行了填充参数的修改（tricks） ② 使用图像修复算法对边缘进行修复，但是效果确实不大好

- **第二种解决方法**——背景不进行平移和缩放，而是使用人像抠图算法抠出人体，只对人体进行平移缩放。这个方法同样需要填充，因为背景抠出人像后原本人像的地方就空了一块。这里的话我试过用图像修复算法进行填充额，但是由于填充面积太大所以效果不好，最后我评估了背景复杂程度之后用knn的方法进行了填充，效果还不错

  



### 比赛中的tricks和疑问

- **分难度进行训练**。由于比赛给了三个难度的数据集，不同难度之间清晰度姿势变化等等差别很大，只用一个模型训练所有难度的话由于高难度数据的影响导致模型在低难度数据上表现也不够好。因此我将简单难度的单独训练一个，简单和中等又训练一个，中等和困难又训练一个。

- **相似表情**。考虑到要生成的人脸和输入的人脸的表情可能很相似的情况，这种情况下就不需要经过模型生成了，而是直接拿输入的人脸即可。实际实现上面就是比较原人脸关键点图和目标图关键点图的L1，设置一个阈值，小于阈值则直接复制输入的人脸。

- **颜色迁移**。首先这是因为我的模型确实存在问题，导致模型生成的人脸与原人脸存在色差。多次尝试改变模型也没有改变，所以最后我就用了传统的颜色迁移算法将原人脸的颜色分布迁移到生成的人脸上，从而缓解颜色偏差的问题。

  

## 其他尝试

比赛过程中也有很多失败的尝试，这部分的代码在`discarded`中，这里只记录几个比较大的尝试。

- 扩大卷积核。做这一尝试的原因是当生成侧脸的时候总是有重影等情况，所以我考虑是不是感受野比较小难以捕捉较远的联系，于是就对卷积核扩大。但是最终没看出有什么改变，就放弃了。
- 关键点判别器。我上面说到关键点检测器的时候说了目前的关键点损失是L1损失，存在很多缺点，所以就想着将关键点的判别工作同样交给判别器做，这个当时好像是因为训练困难以及效果实在太差就很快放弃了。
- 用GAN训练背景融合。背景融合的问题我在上面已经说了，但是最开始我想的是用同样GAN模型去训练。输入为背景和在目标位置的人脸，希望模型生成真实的人脸在目标位置的整张图像。但是显然我太高估GAN的能力，或者说我太高估stargan架构的能力。反正这个任务生成的结果根本就是两张图贴在一起而已。或许是网络架构不适合这个任务，或者是我训练方式有问题，反正这个方法试了一段时间就开始转向传统方法了。

