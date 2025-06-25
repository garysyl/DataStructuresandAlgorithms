# 大语言模型本地部署指南

## 一、硬件选择与分析

### 1. GPU性能与显存的关系
- **GPU性能**：决定模型推理速度，高性能GPU（如NVIDIA A100/H100）可显著提升处理速度。
- **显存容量**：决定可部署模型的最大规模，显存越大支持的参数量越高。

### 2. 不同硬件方案对比

#### **核显（集成显卡）**
- **优点**：
  - 成本低，无需额外购买独立显卡。
  - 共享系统内存作为显存，最高可扩展至96GB。
- **缺点**：
  - 推理速度较慢（如14B模型约10 tokens/s）。
  - 延迟较高，不适合实时交互场景。
- **适用场景**：轻量级模型（如14B以下）或预算有限的用户。

#### **独显（独立显卡）**
- **优点**：
  - 推理速度快（如RTX 5090运行14B模型可达50+ tokens/s）。
  - 支持高精度计算（FP16/FP32）。
- **缺点**：
  - 显存有限（如RTX 5090为24GB），需选择多卡并行（如A100 80GB x2）部署大模型。
  - 成本高昂。
- **推荐型号**：
  - 消费级：RTX 5090（24GB）、RTX 7000
  - 企业级：A100、H100

#### **Mac设备**
- **优点**：
  - M系列芯片（M1/M2/M3/M4）采用统一内存架构（Unified Memory），可利用全部内存作为显存（如M3 Ultra最高512GB，可本地部署满血deepseek）。
- **缺点**：
  - GPU性能较NVIDIA同价位产品弱（如M4 Max vs RTX 5060）。
  - 高显存价格昂贵（512GB Mac Studio 10万元）。
- **适用场景**：中小型模型（如32B以下）或苹果生态用户。

#### **台式机**
- **优点**：
  - 支持多卡并行。
  - 可升级性强（更换显卡/内存）。
- **缺点**：
  - 不便携，需占用空间。
  - 宿舍限电（通常500W以下）限制高性能配置。

### 3. 笔记本选择建议
- **参考来源**：
  - 笔吧评测室
  - 极客湾

---

## 二、软件工具与平台

### 1. 核心工具推荐
| 工具            | 特点                                         | 适用场景                          |
| --------------- | -------------------------------------------- | --------------------------------- |
| **Ollama**      | 命令行工具，支持Intel GPU（需IPEx-LLM）      | 快速部署轻量级模型（如Llama3 8B） |
| **LM Studio**   | 图形化界面，兼容NVIDIA和Mac                  | 新手友好，界面美观                |
| **HuggingFace** | 模型仓库+训练平台，提供量化模型（INT4/INT8） | 自定义模型微调与部署              |
| **ModelScope**  | 阿里达摩院开源模型库，集成一键部署功能       | 中文场景优化模型（如通义千问）    |

### 2. 部署流程示例（以Ollama为例）
```bash
# 安装Ollama（Linux）
curl -fsSL https://ollama.com/install.sh | sh

# 启动服务
ollama serve

# 列出可用模型
ollama list

# 下载并运行Llama3 8B模型
ollama run llama3:8b
```

### 3. 性能优化策略

- **量化技术**：
  - **INT4量化**：将显存需求降低至1/8（如14B模型从84GB降至10.5GB）。
  - **FP16/FP8**：平衡精度与性能，适合企业级部署。
- **分布式计算**：
  - 使用`vLLM`框架提升推理速度（支持A100/H100）。
  - 多卡并行（如`DeepSpeed`库支持NVLink互联）。

## 三、总结与选择建议
1. **预算有限**：选择核显+INT4量化（如M2 Mac运行32B模型）。
2. **追求速度**：投资独显（如RTX 4090）或企业级GPU（如A100）。
3. **开发需求**：优先Mac（统一内存）或Linux台式机（灵活扩展）。
4. **软件适配**：新手推荐LM Studio，高级用户使用Ollama+HuggingFace。

> **扩展阅读**：
> - [Ollama官方文档](https://ollama.com/docs)
> - [HuggingFace模型库](https://huggingface.co/models)
> - [ModelScope中文模型](https://modelscope.cn/models)
### 补充说明：

- **量化技术**：通过降低模型精度（如FP32→INT4），可大幅减少显存占用且精度损失较小（通常<5%）。显存有限推荐Q4量化（包括IQ4-XXS与Q4-K-M)，具体效果见下表（以千问3 14b为例）：

| Filename               | Quant type | File Size | Split | Description                                                  |
| ---------------------- | ---------- | --------- | ----- | ------------------------------------------------------------ |
| Qwen3-14B-bf16.gguf    | bf16       | 29.54GB   | false | Full BF16 weights.                                           |
| Qwen3-14B-Q8_0.gguf    | Q8_0       | 15.70GB   | false | Extremely high quality, generally unneeded but max available quant. |
| Qwen3-14B-Q6_K_L.gguf  | Q6_K_L     | 12.50GB   | false | Uses Q8_0 for embed and output weights. Very high quality, near perfect, recommended. |
| Qwen3-14B-Q6_K.gguf    | Q6_K       | 12.12GB   | false | Very high quality, near perfect, recommended.                |
| Qwen3-14B-Q5_K_L.gguf  | Q5_K_L     | 10.99GB   | false | Uses Q8_0 for embed and output weights. High quality, recommended. |
| Qwen3-14B-Q5_K_M.gguf  | Q5_K_M     | 10.51GB   | false | High quality, recommended.                                   |
| Qwen3-14B-Q5_K_S.gguf  | Q5_K_S     | 10.26GB   | false | High quality, recommended.                                   |
| Qwen3-14B-Q4_K_L.gguf  | Q4_K_L     | 9.58GB    | false | Uses Q8_0 for embed and output weights. Good quality, recommended. |
| Qwen3-14B-Q4_1.gguf    | Q4_1       | 9.39GB    | false | Legacy format, similar performance to Q4_K_S but with improved tokens/watt on Apple silicon. |
| Qwen3-14B-Q4_K_M.gguf  | Q4_K_M     | 9.00GB    | false | Good quality, default size for most use cases, recommended.  |
| Qwen3-14B-Q3_K_XL.gguf | Q3_K_XL    | 8.58GB    | false | Uses Q8_0 for embed and output weights. Lower quality but usable, good for low RAM availability. |
| Qwen3-14B-Q4_K_S.gguf  | Q4_K_S     | 8.57GB    | false | Slightly lower quality with more space savings, recommended. |
| Qwen3-14B-Q4_0.gguf    | Q4_0       | 8.54GB    | false | Legacy format, offers online repacking for ARM and AVX CPU inference. |
| Qwen3-14B-IQ4_NL.gguf  | IQ4_NL     | 8.54GB    | false | Similar to IQ4_XS, but slightly larger. Offers online repacking for ARM CPU inference. |
| Qwen3-14B-IQ4_XS.gguf  | IQ4_XS     | 8.11GB    | false | Decent quality, smaller than Q4_K_S with similar performance, recommended. |
| Qwen3-14B-Q3_K_L.gguf  | Q3_K_L     | 7.90GB    | false | Lower quality but usable, good for low RAM availability.     |
| Qwen3-14B-Q3_K_M.gguf  | Q3_K_M     | 7.32GB    | false | Low quality.                                                 |
| Qwen3-14B-IQ3_M.gguf   | IQ3_M      | 6.88GB    | false | Medium-low quality, new method with decent performance comparable to Q3_K_M. |
| Qwen3-14B-Q3_K_S.gguf  | Q3_K_S     | 6.66GB    | false | Low quality, not recommended.                                |
| Qwen3-14B-Q2_K_L.gguf  | Q2_K_L     | 6.51GB    | false | Uses Q8_0 for embed and output weights. Very low quality but surprisingly usable. |
| Qwen3-14B-IQ3_XS.gguf  | IQ3_XS     | 6.38GB    | false | Lower quality, new method with decent performance, slightly better than Q3_K_S. |
| Qwen3-14B-IQ3_XXS.gguf | IQ3_XXS    | 5.94GB    | false | Lower quality, new method with decent performance, comparable to Q3 quants. |
| Qwen3-14B-Q2_K.gguf    | Q2_K       | 5.75GB    | false | Very low quality but surprisingly usable.                    |
| Qwen3-14B-IQ2_M.gguf   | IQ2_M      | 5.32GB    | false | Relatively low quality, uses SOTA techniques to be surprisingly usable. |
| Qwen3-14B-IQ2_S.gguf   | IQ2_S      | 4.96GB    | false | Low quality, uses SOTA techniques to be usable.              |
| Qwen3-14B-IQ2_XS.gguf  | IQ2_XS     | 4.69GB    | false | Low quality, uses SOTA techniques to be usable.              |

由此可见，bf16的量化需要约30GB显存，但是Q4量化只需要约10GB并且有着类似的效果，这使得我们可以在16GB显存的核显轻薄本或者12GB的5070ti游戏本轻松部署14b小模型。