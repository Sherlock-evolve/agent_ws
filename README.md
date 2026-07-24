# Workspace Agent

一个以安全边界、事务一致性和可测试性为核心的中文工作区 Agent。
项目使用 LangChain 的消息与工具协议，但保留自研 Agent 循环，便于明确控制
流式输出、工具审批、上下文预算、会话持久化、审计和知识检索行为。

## 能力概览

- 流式模型回答和多轮工具调用；
- 受工作区边界保护的文件列出、读取、搜索和原子写入；
- 写入前 diff 审批、审批后文件冲突检测和可恢复审批；
- Agent 循环、工具次数、工具结果和上下文预算；
- 事务式消息提交、长期摘要记忆和命名会话；
- 模型/工具耗时、事件信封和脱敏 JSONL 审计；
- 可选的 `docs/` 语义检索、增量 Embedding 缓存和引用校验；
- `observe` 与 `require-valid` 两种引用策略。

## 运行链路

```text
CLI
  └─ PersistentSession
       ├─ WorkspaceAgent
       │    ├─ Chat Model
       │    └─ ToolExecutionMiddleware
       │         └─ Workspace / Knowledge Tools
       ├─ Session Store
       └─ JSONL Audit Log
```

`contracts.py` 定义跨界事件和审批协议；CLI 只负责输入与事件渲染。
只有模型成功生成最终回答后，本轮消息和长期摘要才会提交。

## 环境要求

- Python 3.10 或更高版本；
- OpenAI 兼容的聊天模型服务；
- 启用知识库时，需要兼容 OpenAI Embeddings 的服务。

创建虚拟环境并安装锁定依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```

开发模式也可以从 `pyproject.toml` 安装：

```bash
python -m pip install -e ".[dev]"
```

## 配置

在项目根目录创建不纳入 Git 的 `.env`：

```dotenv
ZHIPU_MODEL=your-chat-model
ZHIPU_API_KEY=your-api-key
ZHIPU_BASE_URL=https://your-provider.example/v1

# 仅在启用知识库时需要
EMBEDDING_MODEL=your-embedding-model
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_BASE_URL=https://your-provider.example/v1
```

`EMBEDDING_API_KEY` 和 `EMBEDDING_BASE_URL` 未配置时，会分别回退到聊天模型的
`ZHIPU_API_KEY` 和 `ZHIPU_BASE_URL`。启用知识库可能把文档分块和查询发送给
所配置的外部 Embeddings 服务。

## 使用

普通模式：

```bash
python main.py
python main.py --session project-a
```

启用知识库：

```bash
python main.py --enable-knowledge
python main.py \
  --enable-knowledge \
  --knowledge-directory docs \
  --citation-policy require-valid
```

CLI 命令：

```text
:session              显示当前会话状态
:sessions             列出会话
:switch <session_id>  切换或新建会话
:delete <session_id>  删除非当前会话
:pending              显示等待恢复的审批
:resume               恢复并重新确认审批
:retry                重试保存 dirty 会话
:help                 显示帮助
exit / quit / 退出    退出
```

等待审批时退出不会执行工具。下次打开同名会话后，使用 `:resume` 重新生成预览
并再次确认。批准记录会在副作用执行前清除，因此崩溃恢复遵循至多一次语义：
不会凭旧审批重复执行写入，但极端情况下可能需要用户重新发起未执行的操作。

## 测试

```bash
python -m pytest -q
python -m compileall -q .
```

测试使用确定性脚本模型和受控 Embeddings，默认不访问网络。

## 安全边界

- 文件工具只接受工作区内相对路径，拒绝绝对路径、`..` 和符号链接逃逸；
- `.env`、密钥、会话、审计和知识索引目录不会暴露给模型文件工具；
- 写入使用同目录临时文件、`fsync()` 和 `os.replace()`；
- 审批预览与执行之间使用文件 SHA-256 检测冲突；
- 工具事件和审计记录不会保存工具结果正文，敏感参数会被脱敏；
- 只读工具可配置超时并在取消后停止等待；
- 写入等副作用一旦开始会完成原子执行边界，取消只阻止尚未开始的操作；
- 检索内容始终被标记为不可信资料，不能提供系统权限或工具授权。

## 主要目录和状态文件

```text
.agent_sessions/   会话快照和等待恢复的审批，权限 0700/0600
.agent_audit/      每会话 JSONL 审计日志
.knowledge_index/  按模型配置隔离的增量 Embedding 缓存
docs/              默认知识文档目录
tests/             确定性单元和集成测试
```

这些运行时目录均被 Git 忽略，也被工作区文件工具屏蔽。

## 开发路线

详细里程碑、已完成能力和后续服务化方向见
[`docs/agent-development-roadmap.md`](docs/agent-development-roadmap.md)。
