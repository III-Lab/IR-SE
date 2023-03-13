# 红外图像 -> 轻量化网络



## Introduce

```mermaid

flowchart LR
    A(红外图像)
    B(分成8*8的块)
    C(对块进行DCT变换)
    D(将不同的频率放入不同的通道中)
    E(通道注意力机制)
    F(关键通道选择)
    G(DCT反变换)
    H(轻量化神经网络)
    
    A-->B-->C-->D
    D-->E-->F-->G-->H

```

## Result

```mermaid
graph LR
		subgraph Input
		A[<img src='/figures/IR.bmp' width='100' height='100'>]
		end
		subgraph Output
		C[<img src='/figures/result.png' width='100' height='100'>]
		end
    subgraph ChannelSpatialSELayer
        B1[<img src='/figures/DCT.bmp' width='100' height='100'>]
        B2[<img src='/figures/SE.bmp' width='100' height='100'>]
        B3[<img src='/figures/IDCT.bmp' width='100' height='100'>]
    end
    A-->|DCT|B1
   	B1-->|CSSE|B2-->|IDCT|B3
		B3-->|IR-Net|C

```

## Refer

- [squeeze_and_excitation](https://github.com/ai-med/squeeze_and_excitation)
