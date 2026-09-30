# ngn-fba-ai

OCR、LayoutLMv3 和字符清洗的统一仓库，三个子包保留独立依赖环境。
包含训练、评估和三个独立 gRPC GPU 服务；推理协议位于 `proto/ngn_fba_ai_proto/ai.proto`。

| 目录 | 来源 | 用途 |
| --- | --- | --- |
| `packages/ocr/ngn_fba_ocr` | nga-ocr `ab1dcab` | PaddleOCR 预训练模型，无需训练 |
| `packages/layout/ngn_fba_layout` | nga-torch `5d0556a` | LayoutLMv3 训练、推理 |
| `packages/clean/ngn_fba_clean` | nga-clean `22ec63c` | 快递号、SKU、收件人 LSTM 训练、推理 |
| `training` | nga-torch 恢复训练工具 | 数据隔离、五折验证、最终拟合、共同留出集对照 |

主机信息仅放本地且已忽略的 `AGENTS.md`。业务数据、图片和密钥不提交 Git。
发布权重压缩包通过 Git LFS 管理，Git 中只提交指针及 SHA256 校验文件。
开发代码放 `~/projects/ngn-fba-ai`；正式运行形态放 `~/opt/ngn-fba-ai`。

## 训练环境

GPU 环境使用 Python 3.12，各包独立 uv venv，不修改主机 Python/CUDA。
训练锁文件来自已验证的 GPU 环境；推理容器不需要完整训练依赖。

```sh
uv venv --python 3.12 packages/layout/.venv
uv pip sync --python packages/layout/.venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  packages/layout/requirements-training.lock.txt
uv venv --python 3.12 packages/clean/.venv
uv pip sync --python packages/clean/.venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  packages/clean/requirements-training.lock.txt
```

## 最终拟合

不按外层测试最高分选某折权重。读取五折内层选中的最佳训练轮数，取中位数，
再合并开发集重新训练。原时间测试组和 LLM 留出组均排除，不搜索测试集上的阈值。
清洗训练继续排除同一 raw 对应多个相互冲突目标的监督。

```sh
python3 training/test_final_fit.py
python3 training/final_fit.py \
  --data "$FBA_DATA_ROOT" --output "$FBA_DATA_ROOT/final-fit-v1" \
  --layout-python "$PWD/packages/layout/.venv/bin/python" \
  --clean-python "$PWD/packages/clean/.venv/bin/python" \
  --base-model "$FBA_BASE_MODEL"
packages/clean/.venv/bin/python training/compare_final.py \
  --data "$FBA_DATA_ROOT" --fit "$FBA_DATA_ROOT/final-fit-v1"
```

输出目录必须不存在，防止覆盖实验。`training-plan.json` 在训练前记录选择依据、
样本数、训练源码哈希及固定轮数；完成后生成 `TRAINING_COMPLETE.json`。
清洗输出为每个模型的 `model.ckpt` 和 `vocab.json`；LayoutLM 输出 Hugging Face 模型目录。

2026-09-30 最终训练使用 1807 张开发集图片；固定轮数为 LayoutLM 4、track 2、
SKU 9、recipient 19，seed 42。三个清洗任务分别使用去重、冲突过滤后的
7199、1562、1001 条训练记录。

同一批 204 张时间留出图片、相同已保存 OCR 的结果：

| 指标 | 初始恢复版 | 最终拟合版 |
| --- | ---: | ---: |
| 快递号 | 200 | 202 |
| 收件人 | 173 | 177 |
| SKU（含数量） | 201 | 202 |
| 整单全部正确 | 167（81.86%） | 173（84.80%） |

最终拟合版新增答对 17 单、退步 11 单，净增加 6 单。GPU 推理中位耗时约 38ms，
不包含 OCR、网络及服务排队。2026-09-30 已随 gRPC 迁移部署 final-fit-v1；该结果仍低于历史存储结果，迁移成功不代表准确率超过历史模型。
这批数据的历史生产结果为 195/204；恢复训练仍未达到该结果。
以上仅覆盖可以完整对齐 OCR 的业务复核样本，不能代表完整线上准确率；
时间留出集此前已被查看，并非全新的前瞻性测试。千问比较已按用户要求取消。

候选权重包为 `final-fit-v1-models.tar.gz`（487999318 字节），已压缩 rsync 备份并校验。
SHA256：`b13000580db10796c2fa136c2031583982ba965771b29435a1677fcc4e185d2c`。

Git LFS 路径为 `artifacts/models/final-fit-v1-models.tar.gz`，包含 LayoutLMv3 和
track、SKU、recipient 三个清洗模型，以及训练、评估和文件校验记录。
安装 Git LFS 客户端后，在仓库中执行：

```sh
git lfs install --local
git lfs pull --include='artifacts/models/final-fit-v1-models.tar.gz'
cd artifacts/models
shasum -a 256 -c final-fit-v1-models.tar.gz.sha256
```

只获取代码时使用 `GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:mydansun/ngn-fba-ai.git`。

## gRPC 与部署

`compose.yaml` 在 GPU 机器运行 OCR、LayoutLMv3、三个 LSTM 共用的清洗服务及 frpc。
每个模型容器内部使用 50051，均不发布主机端口。服务要求 `x-api-key` metadata，
图片上限20MiB/2500万像素，清洗每批最多256条、每条512字符；GPU调用串行且队列有界。
OCR 返回处理后 JPEG 与文字/坐标数组；LayoutLM 返回预测结构与可视化；清洗按字段类型批处理。
预测 JSON 保留 scan 的既有整数索引、坐标和存储结构，传输层为 protobuf/gRPC。

业务机运行 `deploy/compose.relay.yaml` 中的 frps，GPU 主动连接业务机443。
Traefik 仅按 SNI `fba-ai.internal` 透传此 TLS 连接，使用项目私有 CA 双向验证和 token；
无需新增 DNS 记录或开放公网 gRPC 端口。业务 Docker 网络 `ngn-fba-ai-relay` 内的
`ai-relay:50052/50053/50054` 分别转发 OCR/LayoutLM/clean。

部署前在 GPU 项目目录执行：

```sh
python3 deploy/provision_tunnel.py --output .secrets --server "$FBA_RELAY_HOST"
# 验证 LFS 权重包后解压，形成 models/layout 和 models/clean。
tar -xzf artifacts/models/final-fit-v1-models.tar.gz models
# 将 .secrets/frps 和 deploy/compose.relay.yaml 放到业务机独立 relay 目录。
# 将同一个 .secrets/api_key 安全配置为 scan 的 AI_API_KEY，不打印或提交。
docker compose build
docker compose up -d
```

正式运行 checkout 放 `~/opt/ngn-fba-ai`；`~/projects` 用于开发。
私有 CA 密钥仅留在 `.secrets/pki`，不分发给业务机。叶证书有效期365天，轮换时为两端
重新签发并更新证书后重启 frps/frpc。协议代码可用 `deploy/generate_protocol.sh` 重新生成，
再运行 `python3 deploy/sync_protocol.py <scan-backend-path>` 同步扫描服务的生成代码。

frp 配置依据 [官方示例](https://github.com/fatedier/frp/tree/v0.71.0/conf)，
TLS 入口使用 [Traefik TCP passthrough](https://doc.traefik.io/traefik/reference/routing-configuration/tcp/tls/)。

### 2026-09-30 部署验证

scan 分支 `grpc-ai-monorepo`（`648d494`）已接入三个 gRPC 服务并切换生产后端。
GPU 正式运行目录为 `~/opt/ngn-fba-ai`，业务机 relay 为 `~/opt/ngn-fba-ai-relay`。
三个 GPU 服务健康检查通过；frpc 重启后恢复连接，三个代理均通过真实调用。
缺失客户端证书被 mTLS 拒绝，错误 API key 被三个服务拒绝。

scan 后端 103 项测试通过。三张历史图片从生产 scan 上传到 OCR、LayoutLM、clean、
MongoDB 和 MinIO 预览全链路成功，单次端到端耗时 3.493 / 2.558 / 2.987 秒
（关闭重复展平，含网络和存储；少量冒烟样本，不作为吞吐量或准确率基准）。
另已验证默认开启展平的真实请求，3.011 秒成功完成三类清洗。
验证记录和 S3 对象均已按本次生成的 ID 删除。旧独立 OCR 容器已停止。
回滚所需的旧后端镜像标签及环境文件保存在业务机私有运维目录，不提交凭据。
