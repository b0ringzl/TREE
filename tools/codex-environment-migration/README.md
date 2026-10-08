# TREE Codex 环境迁移工具

这套脚本只迁移环境配置，不复制 `D:\TREE` 的项目代码、数据集、模型、Git 历史或工作树。

## 包含内容

- Windows、CPU、GPU 和关键工具版本清单。
- Winget 可识别的软件清单。
- Conda 环境的 `history` 与 `exact` 两套 YAML 定义。
- VS Code 扩展、全局 npm 包和系统 Python 包清单。
- 项目中的小型依赖定义文件，例如 `requirements.txt`、`package-lock.json` 和 `environment.yml`。
- 经过便携化处理的 Codex `config.toml`、用户 `AGENTS.md`、规则和非系统技能。

明确不包含 Codex 登录信息、聊天数据库、浏览器状态、插件缓存、Token 或项目文件。

## 一、在旧电脑导出到 U 盘

先退出正在运行的开发任务。假设 U 盘是 `E:`：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& "D:\TREE\tools\codex-environment-migration\Export-TreeEnvironment.ps1" -DestinationRoot "E:\"
```

如果不需要 Winget 软件清单：

```powershell
& "D:\TREE\tools\codex-environment-migration\Export-TreeEnvironment.ps1" -DestinationRoot "E:\" -SkipWingetExport
```

默认不导出 Codex memories。确实需要时显式增加 `-IncludeMemories`，并在恢复时单独增加 `-RestoreMemories`。

## 二、在新电脑预览差异

找到 U 盘中生成的 `TREE-Codex-Environment-日期时间` 文件夹：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& "E:\TREE-Codex-Environment-日期时间\scripts\Restore-TreeEnvironment.ps1" `
  -PackageRoot "E:\TREE-Codex-Environment-日期时间"
```

不加恢复开关时只做安全预览和版本比较。

## 三、建议分阶段恢复

先安装软件：

```powershell
& "E:\TREE-Codex-Environment-日期时间\scripts\Restore-TreeEnvironment.ps1" `
  -PackageRoot "E:\TREE-Codex-Environment-日期时间" `
  -InstallWingetPackages -IgnorePackageVersions
```

重新打开 PowerShell，再恢复 Codex、Git、VS Code 和 Conda：

```powershell
& "E:\TREE-Codex-Environment-日期时间\scripts\Restore-TreeEnvironment.ps1" `
  -PackageRoot "E:\TREE-Codex-Environment-日期时间" `
  -ProjectRoot "D:\TREE" `
  -ApplyCodex -ApplyGitConfig -InstallVSCodeExtensions -CreateCondaEnvironments
```

也可以使用 `-Full` 执行以上主要阶段；memories 仍不会自动恢复。

## Conda 定义选择

- 默认 `History`：只保留显式安装的主要包，更适合新电脑和不同驱动环境。
- `Exact`：包含完整包版本，重现性更高，但可能因 CUDA、Python 或软件源变化而无法求解。

需要精确定义时增加：

```powershell
-CondaDefinition Exact
```

## 重要说明

- 新电脑已经登录同一 Codex 账号，因此脚本不会迁移 `auth.json` 等认证文件。
- 当前机器的 Codex `notify` 和内置 `node_repl` MCP 配置包含应用安装路径或本机管道，导出时会移除，由新电脑上的 Codex 自动重建。
- ArcGIS MCP 配置会把旧路径替换成新 `ProjectRoot`，但其项目内 `.venv` 仍需在项目文件到位后重建。
- 如果配置中检测到 Token、密码或 API Key，导出包只留下 `__REQUIRED_SECRET__`；恢复脚本不会覆盖当前 `config.toml`，而会生成 `config.toml.migration-candidate` 等待人工补齐。
- 恢复 Codex 配置前，脚本会在新电脑的 `.codex` 下创建带时间戳的备份目录。
