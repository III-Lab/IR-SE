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
		subgraph Original-Image
		A(<img src='/figures/IR.bmp'>)
		end
		subgraph Output-Image
		C(<img src='/figures/result.png'>)
		end
		
    subgraph Block-DCTimage&CompressedDCT-Image&Compressed-Image
        	B1(<img src='/figures/DCT.bmp' >)
        	B2(<img src='/figures/SE.bmp'>)
        	B3(<img src='/figures/IDCT.bmp'>)
    end
    A-->|DCT|B1
    B1-->|CSSE|B2-->|IDCT|B3
		B3-->|IR-Net|C

		style Original-Image fill:#ffffff,stroke:#13,stroke-width:1px
		style Output-Image fill:#ffffff,stroke:#23,stroke-width:1px
		style Block-DCTimage&CompressedDCT-Image&Compressed-Image fill:#ffffff,stroke:#33,stroke-width:1px
		
		style A fill:#ffffff,stroke:#fff,stroke-width:0px
		style B1 fill:#ffffff,stroke:#fff,stroke-width:0px
		style B2 fill:#ffffff,stroke:#fff,stroke-width:0px
		style B3 fill:#ffffff,stroke:#fff,stroke-width:0px
		style C fill:#ffffff,stroke:#fff,stroke-width:0px
```

## Refer

- [squeeze_and_excitation](https://github.com/ai-med/squeeze_and_excitation)
