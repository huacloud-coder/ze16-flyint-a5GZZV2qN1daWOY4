

> 论文提出一种新颖的`POA`自监督学习范式，通过弹性分支设计允许同时对多种尺寸的模型进行预训练。`POA`可以直接从预训练`teacher`生成不同尺寸的模型，并且这些模型可以直接用于下游任务而无需额外的预训练。这个优势显著提高了部署灵活性，并有助于预训练的模型在各种视觉任务中取得`SOTA`结果。
> 
> 
> 来源：晓飞的算法工程笔记 公众号


**论文: POA: Pre\-training Once for Models of All Sizes**


![](https://developer.qcloudimg.com/http-save/6496381/6594bbe2db61fc75aacce473f01fd04b.png)


* **论文地址：[https://arxiv.org/abs/2408\.01031](https://github.com)**
* **论文代码：[https://github.com/Qichuzyy/POA](https://github.com):[wgetCloud机场](https://tabijibiyori.org)**


# Abstract




---


  大规模自监督预训练为一个基础模型处理多种不同的视觉任务铺平了道路。大多数预训练方法在一次训练中训练一个特定大小的单一模型。然而，在现实世界的场景中，由于各种计算或存储限制，需要大量的工作来开发一系列不同大小的模型进行部署。因此，在这项研究中，我们提出了一种新颖的三分支自监督训练框架，称为`POA`（`Pre-training Once for All`），来解决上述问题。我们的方法在现代自蒸馏范式中引入了一种创新的弹性`student`分支。在每个预训练步骤中，我们随机从原始`student`中抽样一个子网络来形成弹性`student`，并以自蒸馏的方式训练所有分支。一旦预训练完成，`POA`允许提取不同大小的预训练模型用于下游任务。值得注意的是，弹性`student`促进了多个不同大小模型的同时预训练，同时也作为各种大小模型的额外集合，增强了表示学习。大量实验证明了我们的`POA`的有效性和优势，包括k最近邻、线性探测评估以及多个下游任务的评估。它使用`ViT`、`Swin Transformer`和`ResNet`骨干网络实现了最先进的性能，并通过一次预训练会话生成了大约一百个不同大小的模型。代码可在以下链接找到：`https`😕/`github.com`/`Qichuzyy`/`POA`。


# Introduction




---


  通过自监督学习在大型模型中学习可泛化的视觉表示，近年来在各种视觉任务上取得了卓越的性能。然而，当部署到现实世界的应用程序时，大型模型必须根据计算、存储、功耗等各种资源限制进行调整。例如，一个良好设计的人工智能产品通常包括一套为不同场景量身定制的模型，比如`Gemini Nano`、`Pro`和`Ultra`。对于一个大型预训练模型，将其部署到具有不同资源约束的多个应用场景的常见解决方案包括额外的权重修剪、知识蒸馏，甚至从头开始重新训练一个小网络，这些都需要大量的开发工作。因此，这个问题引发了一个关键问题：是否可能进行一次预训练以同时生成多个具有不同大小的模型，每个模型都提供足够好的表示。


  为了解决这一挑战，论文引入了一种名为`POA`（`Pre-training Once for All`）的新型自监督学习范式。`POA`建立在流行的`teacher-student`自蒸馏框架之上，具有一个额外的创新性弹性`student`分支。弹性`student`分支通过参数共享嵌入了一系列子网络，这是基于观察到对于现代网络结构来说，较小尺寸的模型是较大尺寸模型的子网络。此外，该分支的参数与原始的或完整的`studennt`共享。在每个预训练步骤中，从完整`student`中随机抽样一部分参数，形成相应的弹性`studennt`。原始完整`student`和弹性`student`都被训练以模拟`teacher`网络的输出。`teacher`本身通过对`student`参数的指数移动平均（`EMA`）不断优化，包括采样的弹性`student`。弹性`student`有助于在不同参数子集上进行有效和高效的预训练，从而成功地从预训练`teacher`中提取出高性能子网络，用于后续的下游场景。它还作为一种训练正则化形式，通过强制`teacher`和各种子网络之间的输出匹配来促进稳定的训练过程。


![](https://developer.qcloudimg.com/http-save/6496381/7bf5450eabe13cdf9376c4a5dfdc7ec3.png)


  `POA`代表了第一个能够同时训练多个不同尺寸模型的自监督学习方法，每个模型在不需要进一步预训练的情况下，都能获得适用于不同资源约束的高质量表示。图`1`显示了通过`POA`预训练的`ViT-L`模型提取的`143`个子网络的`k`最近邻（`k-NN`）评估结果。通过选择不同的弹性宽度和深度，预训练`teacher`模型可以根据可用计算资源定制的适用于下游应用的合适模型，生成足够数量的候选子网络以供选择。值得注意的是，由于在同视图蒸馏上进行了精心设计，每个子网络都经过了良好训练，并表现出优越性能。特别是，`ViT-S`、`ViT-B`和`ViT-L`模型创造了新的基准，与那些由现有方法预训练的模型相比取得了`SOTA`结果。


  为了严格评估方法的有效性，使用三种广泛使用的骨干架构，即`ViT`、`Swin Transformer`和`ResNet`，进行了大量实验。每个骨干架构都在`ImageNet-1K`数据集上进行了预训练，并使用`k-NN`和线性探测分类评估，以及在下游密集预测任务进行评估，如目标检测和语义分割。`POA`在单次预训练会话中跨多种模型尺寸实现了最先进的准确性。


  本文的技术贡献总结如下：


1. `POA`是第一个将无监督表示学习和一次性模型生成集成到单个预训练会话中的预训练范式，解决了社区很少探讨的一次性预训练挑战。这对实际部署非常重要，因为实际部署通常需要一套模型。
2. 提出了一个新颖而优雅的组件，称为弹性`student`（`Elastic Student`），具有一系列弹性算子，可以使`POA`与包括`ViT`、`Swin Transformer`和`ResNet`在内的流行骨干结构兼容，具备生成各种大小模型的能力。此外，还作为模型集成来平滑训练过程并改善学到的表示。
3. 通过对`k-NN`、线性探测和下游密集任务评估的彻底评估，在多个指标上展现出优于现有最先进预训练方法的性能。此外，将`POA`与自监督蒸馏（`SEED`）进行了比较，`SEED`是一种专为自监督学习设计的知识蒸馏方法，进一步验证了`POA`的有效性。


# POA Self\-supervised Learning Framework




---


![](https://developer.qcloudimg.com/http-save/6496381/9c71da51caccd739182d5202080390e5.png)


  论文的主要目标是通过单次自监督预训练会话来预训练多种规模的模型，受到自蒸馏技术最新进展的启发,提出了一个名为`POA`的新型`SSL`（`Self-supervised Learning`）框架。`POA`架构如图`2`所示，包括一个`teacher`模型、一个完整的`student`模型、一个弹性`student`模型以及对应的头部。`teacher`模型使用`student`模型的指数移动平均（`EMA`）进行更新。弹性`student`模型是完整`student`模型的派生版本，其主干网络和头部参数是共享的。


  在两个方面利用蒸馏技术：完整`student`和弹性`student`都是通过使用同一图像不同视图的`teacher`模型进行蒸馏，而弹性`student`还通过使用相同视图的完整`student`进行学习。交叉视图蒸馏作为一种表示学习形式，如所介绍的那样。值得注意的是，除了仅使用完整`student`进行常规`EMA`更新外，弹性`student`在每个预训练步骤中还提供一个随机抽样的子网络，参与`teacher`模型的`EMA`优化。这个过程实际上模拟了多个子网络的集成，这在监督学习领域也被证明是有益的。同视图蒸馏是完整`student`和弹性`student`之间的标准知识蒸馏，提升了弹性`student`的质量。


## Design of Elastic Student


  弹性`student`是一个子网络，其参数是从完整`student`中提取的。在`transformer`主干网络的背景下，宽度指的是标记的维度，而在卷积主干网络中，宽度表示通道数。深度则定义为`transformer`或卷积网络中基本块的数量。给定宽度和深度的值，会产生一定的网络结构。为简单起见，论文将重点放介绍`ViT`的弹性设计。


![](https://developer.qcloudimg.com/http-save/6496381/db0406eb07dc9dc4d3aa5ed23bfdd8bf.png)


  `ViT`的基本块主要由多头自注意力（`MSA`）模块和多层感知器（`MLP`）模块组成。在每个模块之前应用层归一化（`LN`），并在每个模块后使用残差连接。如图`3`的左侧所示，弹性块是指在`ViT`原始基本块中调整宽度后堆叠的弹性`MSA`、`MLP`和`LN`。在论文的方法中，弹性`student`分支是通过在每个训练迭代中组装特定数量的这些弹性块来构建的。


* ### Elastic MSA


  一个原始或完整的`MSA`模块由三个主要组件组成，即输入投影层，包含注意力和连接的操作符，以及输出投影层。将投影层定义为( w∗,b∗w∗,b∗ )，其中 w∗ 表示线性转换权重， b∗ 表示相应的偏置， ∗ 表示层的名称。如图`3`的右侧所示，给定一个标记维度 Dmax\=Nh⋅Dh ，其中 Nh 是注意力头的数量， Dh 是头部维度，具有长度 T 的输入序列 z∈RT×Dmax 最初被投影以形成查询 Q∈RT×Dh 、键 K∈RT×Dh 和值 V∈RT×Dh 。为了生成弹性`MSA`，定义了`M+1`个弹性宽度，包括 Dmax ，间隔为 Dh：


Di\=(Nh−i)⋅Dh,∀i∈{0,1,...,M},M\<Nh.  对于每个弹性宽度 Di ，从完整`MSA`中的相应输入投影层( wa1 , ba1 )中提取生成每个头部的 Q 、 K 和 V 的权重 wa1i∈RDh×Di 和偏置 ba1i∈RDh ，如 wa1i\=wa1\[:,:Di]⋅αi 和 ba1i\=ba1 。这里， αi 表示用于应对输入维度减少的缩放因子，计算公式为 αi\=Dmax/Di 。随着宽度的减小，弹性`MSA`中的注意力头数量自然减少到 Nh−i 。类似地，对于输出投影层( wa2 , ba2 )，权重 wa2i∈RDi×Di 和偏置 ba2i∈RDi 被提取为：


wa2i\=wa2\[:Di,:Di]⋅αi     ba2i\=ba2\[:Di].* ### Elastic MLP


  `ViT`块中的原始或完整`MLP`模块包含两个投影层。第一层( wm1,bm1 )将嵌入维度扩展了 s 倍，通常在`ViT`结构中设置为`4`。然后，第二层( wm2,bm2 )将其投影回原始维度。弹性`MLP`的两个层的参数以类似于公式`2`描述的方式提取，如下所示：


wm1i\=wm1\[:Di⋅s,:Di]⋅αi     bm1i\=bm1\[:Di⋅s]wm2i\=wm2\[:Di,:Di⋅s]⋅αi     bm2i\=bm2\[:Di].* ### Elastic LN


  对于弹性`LN`，直接使用原始`LN`内部参数的前 Di 个元素，类似于公式`2`中的偏置提取。


* ### Elastic depth


  要从包含 Lmax 个块的完整`ViT`中创建一个包含 Li 个弹性块的子网络，引入了一组`N+1`个弹性深度，定义为 Li\=Lmax−i,  ∀i∈{0,1,...,N},  N\<Lmax 。对于特定深度 Li ，根据块`ID`在等间隔上选择相应的块。激活深度 Li 的每个块`ID` BIDLij 可以表示为：


BIDLij\=⌊(Lmax−1)⋅jLi−1⌋,∀j∈{0,1,...,Li−1}.  因此，通过结合弹性宽度和深度，可以生成总共 (N\+1)⋅(M\+1) 个不同的子网络。例如，通过将弹性宽度设置为`384`，弹性深度设置为`12`，可以直接从如`ViT-L`的完整网络中提取一个`ViT-S`。在预训练的每次迭代中，随机选择其中一个子网络作为弹性`student`分支。


## Distillation between Views


  `POA`根据其三个分支执行蒸馏。给定输入图像 x 的一对全局增强视图，表示为 xa 和 xb ，`teacher`编码器 ET 使用 xa 作为输入提取特征 Za\=ET(xa) 。同时， xb 被输入到完整`student`编码器 EIS 和弹性`student`编码器 EES 中，分别产生特征 Zb1\=EIS(xb) 和 Zb2\=EES(xb) 。从`teacher`编码器输出的特征 Za 经过`teacher`头部 HT 处理，然后使用`Sinkhorn-Knopp`（`SK`）算法进行居中处理，并使用温度缩放`softmax`进行归一化，生成概率 pa ，如下所示：


la\=SK(HT(Za)), la∈RP  pia\=exp(lia/τ)∑P−1k\=0exp(lka/τ), ∀i∈{0,...,P−1},  其中 P 是原型（`logits`?）的数量， τ\>0 是温度参数。类似地，通过使用`student`头部 HIS 和 HES 处理输出来计算完整和弹性`student`编码器的概率 pib1 和 pib2 。然后，这些输出通过一个针对`student`量身定制的温度参数 τ′ 的温度缩放`softmax`函数进行处理。值得注意的是， HIS 和 HES 共享相同的参数，只是 HES 的第一个投影层进行公式`2`的相应调整，以便对齐相应的维度。为简单起见，省略了 pib1 和 pib2 的显式表达式，因为它们遵循与公式`5`类似的计算方式。对于完整`student`分支，使用跨视图数据从`teacher`进行蒸馏如下：


LgIS\=−palog(pb1).  弹性`student`分支在`POA`框架中发挥着至关重要的作用。为了确保这一分支的充分训练，采用了从`teacher`和完整`student`分支进行的双重蒸馏。第一次蒸馏涉及到`teacher`模型，利用跨视图数据来引导表示学习。第二次是与完整`student`模型进行的蒸馏过程，使用同视图数据。这种同视图蒸馏负责将完整`student`学到的表示转移到弹性`student`分支。这种双重蒸馏过程的损失函数制定如下


LgES1\=−palog(pb2),LgES2\=−pb1log(pb2).  请注意，在这两个损失函数中，对所有原型求和，以计算相应概率分布之间的交叉熵损失。


## Overall Loss of POA


  根据`SSL`方法，采用多裁剪策略从单个图像中创建各种失真视图。除了之前提到的两个全局视图外，还生成 v 个分辨率较低的局部视图 xl1,xl2,...,xlv 。这些局部视图由两个`student`共同处理，以促进局部到全局的对应关系。完整和弹性`student`的局部蒸馏损失计算如下：


LlIS\=−1vv∑i\=1palog(pli1),LlES1\=−1vv∑i\=1palog(pli2),LlES2\=−1vv∑i\=1pli1log(pli2),  其中， pli1 和 pli2 分别是完整和弹性`student`分支对于局部视图 li 产生的概率。完整和弹性`student`的总蒸馏损失通过将它们与因子 λ 相加来计算：


LS\=λ(LgIS\+LlIS)\+(1−λ)((LgES1\+LlES1)\+(LgES2\+LlES2))\=λLIS\+(1−λ)(LES1\+LES2).  为了确保弹性`student`的每个子网络都得到充分的训练，在主干网络之后引入了多个投影头（`MPH`）。每个投影头具有完全相同的结构，只是原型数量不同。对于每个投影头，根据公式`10`计算完整和弹性`student`的蒸馏损失 LSi 。最终，在具有 H 个投影头的`POA`框架中，整体损失函数被表述为： L\=1H∑Hi\=1LSi 。


# Experiments




---


![](https://developer.qcloudimg.com/http-save/6496381/1c9688d50addb5505d3fdaeea8be48e2.png)


![](https://developer.qcloudimg.com/http-save/6496381/f5909e0f73a5a7f7f49db038129b4def.png)


![](https://developer.qcloudimg.com/http-save/6496381/243463b51dd1d9e779a1f57346c3609d.png)


![](https://developer.qcloudimg.com/http-save/6496381/c27ba44514d58a2f9832d4bf7a20f470.png)


![](https://developer.qcloudimg.com/http-save/6496381/4c5e8aa83735427b101549128ac7d43a.png)


![](https://developer.qcloudimg.com/http-save/6496381/dca59947299de0d29105910b268b066e.png)


![](https://developer.qcloudimg.com/http-save/6496381/944776d7f21bcf67e419bae1589d639f.png)


 
 
 



> 如果本文对你有帮助，麻烦点个赞或在看呗～
> 更多内容请关注 微信公众号【晓飞的算法工程笔记】


![work-life balance.](https://upload-images.jianshu.io/upload_images/20428708-7156c0e4a2f49bd6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
