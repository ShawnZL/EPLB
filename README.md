为保证MoE部分不同专家之间的负载均衡，会将共享专家和高负载的细粒度专家在集群的不同GPU做多个复制，让GPU把更多的热数据（发给共享专家的）跑起来。这个方案就是EPLB。

﻿

EPLB具有如下核心特点：

1）**负载均衡优化**

EPLB 通过复制高负载专家（Redundant Experts Strategy）并对专家分配进行启发式调整，确保不同 GPU 之间的负载均衡。这种方法解决了专家并行中因专家负载不均导致的计算资源浪费问题。分层负载平衡策略也可用于预填充阶段，具有较小的专家并行规模。

**分层负载均衡** 当服务器节点数除以专家组数时，我们使用分层负载平衡策略来控制组受限专家路由。我们首先将专家组均匀地打包到节点，确保不同节点的负载平衡。然后，我们在每个节点内复制专家。最后，我们将复制的专家打包到各个 GPU，以确保不同 GPU 的负载平衡。**分层负载平衡策略可用于预填充阶段，具有较小的专家并行规模**。

**全局负载均衡** 在不同GPU内进行负载均衡，适合大规模的专家并行

2）**跨节点通信优化**

在 DeepSeek-V3 的技术报告中提到，**EPLB 尝试将同一组的专家尽量分配到同一节点，减少跨节点的数据传输开销**。这种分组限制路由（Group-Limited Expert Routing）策略显著提升了分布式训练的效率。

3）**高效可扩展性**

The following code illustrates an example of a two-layer MoE model, and each layer contains 12 experts. We introduce 4 redundant experts per layer, and the total 16 replicas are placed on 2 nodes, and each node contains 4 GPUs. 12个专家，4个多余的专家在每一层，每一层都是1-12（1 4 5 10）（1 5 6 8）

﻿

先计算逻辑专家分配，然后将结果返回再分发下去物理GPU参数。

```go
# Output:
# tensor([[ 5,  6,  5,  7,  8,  4,  3,  4, 10,  9, 10,  2,  0,  1, 11,  1],
#         [ 7, 10,  6,  8,  6, 11,  8,  9,  2,  4,  5,  1,  5,  0,  3,  1]])
```

![](example.png)

# 代码解释

用于在分布式环境中优化混合专家模型(MoE, Mixture of Experts)的部署。我将逐函数详细解释其工作原理。

## 1. `balanced_packing` 函数

```python
def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
```

### 功能

将n个带权重的对象分配到m个包中，使每个包恰好包含n/m个对象，并且所有包的权重尽可能平衡。

### 参数

- ﻿`weight`: [X, n] 形状的张量，表示每个项目的权重
- ﻿`num_packs`: 包的数量

### 返回值

- ﻿`pack_index`: [X, n] 形状的张量，表示每个项目的包索引
- ﻿`rank_in_pack`: [X, n] 形状的张量，表示项目在包中的排名

### 算法逻辑

1. 如果每pack只包含一个组(`groups_per_pack=1`)，直接返回简单的分配，每一个rank都是0
2. 否则，将项目按权重从大到小排序
3. 对于每个排序后的项目，将其分配到当前权重最小且未满的包中
4. 更新包的权重和项目计数

这是一种贪心算法，通过始终选择当前权重最小的包来实现负载平衡。

## 2. `replicate_experts` 函数

```python
def replicate_experts(weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

### 功能

将`num_log`个逻辑专家复制到`num_phy`个物理副本，使所有副本的最大负载最小化。

### 参数

- ﻿`weight`: [X, num_log] 形状的张量，表示每个逻辑专家的负载
- ﻿`num_phy`: 复制后的物理专家总数

### 返回值

- ﻿`phy2log`: [X, num_phy] 形状的张量，每个物理专家对应的逻辑专家ID
- ﻿`rank`: [X, num_phy] 形状的张量，每个物理专家的副本排名
- ﻿`logcnt`: [X, num_log] 形状的张量，每个逻辑专家的副本数量

### 算法逻辑

1. 初始化每个逻辑专家有一个物理副本
2. 对于剩余的物理专家槽位，每次选择当前"平均负载"(负载/副本数)最高的逻辑专家进行复制
3. 更新被选中的逻辑专家的副本计数

这是一种贪心算法，通过不断为负载最高的专家添加副本来平衡系统。

### 代码解释

1. ﻿`**weight / expertcnt**`:

- - 这是一个逐元素的除法操作，将`weight`张量中的每个元素除以`expertcnt`张量中的对应元素。
  - 结果是一个新的张量，表示每个逻辑专家的“平均负载”或“单位副本的负载”。

1. ﻿`**.max(dim=-1)**`:

- - ﻿`dim=-1`指的是沿着最后一个维度（即每一行）进行操作。
  - ﻿`max`函数返回两个值：
    - 最大值：每一行的最大元素值。
    - 最大值的索引（indices）：每一行中最大值所在的索引。

1. ﻿`**.indices**`:

- - 选择了`max`函数的返回结果中的第二个值，即每一行中最大值的索引。
  - ﻿`redundant_indices`是一个一维张量，形状为`[num_layers]`，其中的每个元素是一个整数，表示该层中负载最大的逻辑专家的索引。

### 返回值示例

假设：

- ﻿`weight`是一个`[num_layers, num_experts]`的张量。
- ﻿`expertcnt`是同样形状的张量，表示每个逻辑专家的副本数量。

例如：

```plain
weight = torch.tensor([[10.0, 20.0, 30.0],
                       [15.0, 25.0, 35.0]])

expertcnt = torch.tensor([[1, 2, 1],
                          [2, 1, 1]])
```

计算`weight / expertcnt`得到：

```plain
[[10.0, 10.0, 30.0],
 [7.5, 25.0, 35.0]]

```

然后，`.max(dim=-1).indices`计算结果为：

```plain
[2, 2]
```

这意味着：

- 在第一个层（第一行），最大值`30.0`出现在索引`2`的位置。
- 在第二个层（第二行），最大值`35.0`出现在索引`2`的位置。

因此，`redundant_indices`返回的值是`[2, 2]`，表示在每一层中需要复制的逻辑专家的索引。

```go
rank[:, i] = expertcnt[arangen, redundant_indices]
```

### 张量的结构

- ﻿`**arangen**`: 是一个一维张量，包含从 `0` 到 `num_layers-1` 的整数。这表示行索引。

```python
arangen = torch.arange(num_layers, dtype=torch.int64)
```

- ﻿`**redundant_indices**`: 是一个一维张量，包含每一行中需要复制的逻辑专家的索引。
- ﻿`**expertcnt**`: 是一个二维张量，形状为 `[num_layers, num_experts]`，存储每个逻辑专家的副本数量。

### 索引操作

在 PyTorch 中，使用高级索引时，`arangen` 和 `redundant_indices` 的组合用于同时选择 `expertcnt` 中的行和列。

```python
expertcnt[arangen, redundant_indices]
```

- **解释**:
  - ﻿`arangen` 提供了行索引。
  - ﻿`redundant_indices` 提供了列索引。
  - 这两者结合使用时，每个索引对 `arangen[i]` 和 `redundant_indices[i]` 将选择 `expertcnt` 中的一个元素。

### 实际计算

假设：

- ﻿`num_layers = 3`﻿
- ﻿`num_experts = 4`﻿
- ﻿`expertcnt` 的初始值为：

```plain
[[1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1]]
```

- ﻿`redundant_indices = [2, 0, 3]`﻿

那么 `expertcnt[arangen, redundant_indices]` 的计算结果为：

- ﻿`arangen = [0, 1, 2]`﻿

索引操作 `expertcnt[arangen, redundant_indices]` 将选择：

- ﻿`expertcnt[0, 2]` -> 取第 0 行第 2 列的元素（值为 1）
- ﻿`expertcnt[1, 0]` -> 取第 1 行第 0 列的元素（值为 1）
- ﻿`expertcnt[2, 3]` -> 取第 2 行第 3 列的元素（值为 1）

结果是 `[1, 1, 1]`，这就是 `rank[:, i]` 被赋值的内容。

### 总结

因此，`expertcnt[arangen, redundant_indices]` 使用高级索引机制，通过 `arangen` 和 `redundant_indices` 的组合来选择 `expertcnt` 中的特定元素，并将这些元素赋值给 `rank` 张量的相应位置。

## 3. `rebalance_experts_hierarchical` 函数

```python
def rebalance_experts_hierarchical(weight: torch.Tensor, num_physical_experts: int, 
                      num_groups: int, num_nodes: int, num_gpus: int):
```

### 功能

实现层次化的专家平衡策略，考虑了节点内和节点间的网络拓扑。

### 参数

- ﻿`weight`: [num_moe_layers, num_logical_experts] 形状的张量，表示每个逻辑专家的负载
- ﻿`num_physical_experts`: 复制后的物理专家总数
- ﻿`num_groups`: 专家组的数量
- ﻿`num_nodes`: 服务器节点数量
- ﻿`num_gpus`: GPU数量

### 返回值

- 物理到逻辑的映射
- 物理专家的排名
- 每个逻辑专家的副本计数

### 算法逻辑

1. 首先定义辅助函数`inverse`用于计算排列的逆
2. 步骤1: 将专家组打包到节点上，使用`balanced_packing`算法
3. 步骤2: 在节点内构建冗余专家，使用`replicate_experts`算法
4. 步骤3: 将物理专家打包到GPU上，再次使用`balanced_packing`算法
5. 整合所有映射，生成最终的物理到逻辑的映射关系

这是一种层次化的负载平衡策略，考虑了分布式系统中的网络拓扑，优先在节点内分配专家以减少跨节点通信。

## 4. `rebalance_experts` 函数

```python
def rebalance_experts(weight: torch.Tensor, num_replicas: int, num_groups: int,
                     num_nodes: int, num_gpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

### 功能

专家并行负载平衡器的入口点函数。

### 参数

- ﻿`weight`: [layers, num_logical_experts] 形状的张量，所有逻辑专家的负载统计
- ﻿`num_replicas`: 物理专家数量
- ﻿`num_groups`: 专家组数量
- ﻿`num_nodes`: 服务器节点数量
- ﻿`num_gpus`: GPU数量

### 返回值

- ﻿`physical_to_logical_map`: [layers, num_replicas] 形状的张量，每个副本的专家索引
- ﻿`logical_to_physical_map`: [layers, num_logical_experts, X] 形状的张量，每个专家的副本索引
- ﻿`expert_count`: [layers, num_logical_experts] 形状的张量，每个逻辑专家的物理副本数量

### 算法逻辑

1. 根据专家组数量是否为节点数量的倍数，选择不同的负载平衡策略:

- - 如果是倍数关系，使用层次化负载平衡策略
  - 否则使用全局负载平衡策略

1. 构建逻辑专家到物理专家的映射

## 总体设计思路

这个负载平衡器的设计针对混合专家模型(MoE)在分布式环境中的部署优化，主要考虑了以下几个方面:

1. **负载平衡**: 确保各个物理设备(GPU)上的计算负载尽可能均衡
2. **网络拓扑感知**: 考虑节点内通信(如NVLink)比节点间通信更快的特性
3. **专家复制**: 通过复制高负载专家来分散计算压力
4. **层次化分配**: 先将专家组分配到节点，再在节点内分配到GPU，减少跨节点通信

# 疑惑

为什么全局部分就是直接复制这部分专家

```
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = replicate_experts(weight, num_replicas)
```

