本项目已使用uv创建python环境.venv，并且可以使用build_wheel.sh构建库到环境里，rspandas是使用rust开发和pandas相同方法的python库，python环境已安装pandas 可以借鉴一下方法具体实现

git commit -am "release: v0.1.8"
git tag v0.1.8

git push origin main --tags

# 列出本地所有的tag

git tag | grep '^v'

# 删除 本地所有的tag

git tag | grep '^v' | xargs git tag -d

# 优化方向

## 优化python层和rust层

1. 优化python层和rust层， python层提供方法接口，rust层底层实现

针对项目本身，这里有一些具体的优化方向和切入点，供你参考：

2. 核心逻辑：进一步“下沉”到 Rust

这是提升性能最根本的手段，目标是将更多 Python 层的循环和控制逻辑移到 Rust 中。

- 复杂绘图类型：像 fill_between、stackplot 这类需要大量计算的函数，可以完全在 Rust 端实现。
- 数据预处理：将归一化、插值、坐标变换等操作在 Rust 中完成，只将结果传回 Python。
- python层引用rsnumpy库，避免在rust层实现numpy相关函数。

3. 提升编码速度

JPEG 编码你用的是 jpeg-encoder，它的速度通常不如 image crate 的 jpeg 编码器（基于 mozjpeg 或 libjpeg-turbo）。可以考虑切换到 image 后端，利用其优化的编码器。

但你的 Cargo.toml 中已有 jpeg-decoder，如果仅用于解码（如加载背景图），那没问题。

4. 边界效率：优化 PyO3 绑定， Python 和 Rust 之间的数据转换是主要开销，需要重点优化。

- 利用零拷贝：使用 PyO3 的 numpy 集成（PyArray）传递大数据，避免复制。
- 减少类型转换：在 Rust 内部尽量使用原生类型（如 ndarray），避免在 Python 对象和 Rust 结构体间频繁转换。
- 批量操作：对外提供接收数组的 API，在 Rust 内部一次性完成循环，你已经在 scatter 中这么做了，可以推广到更多函数。
- 启用 LTO：在 Cargo.toml 中为 release 启用链接时优化（LTO），能减少 FFI 边界开销。

5. 渲染后端：挖掘 plotters 潜力

- 升级与调参：保持 plotters 为最新版本，并尝试调整其后端（如 BitmapBackend）的渲染参数。
- 探索替代后端：如果默认后端是瓶颈，可研究 plotters-conrod 等更高效的后端。
- 算法优化：研究并实现线段简化（Line simplification）等技术，在保证视觉质量的同时减少渲染负担。

6. 并发与内存：释放 Rust 能力

- 内存池：考虑在 Rust 端实现内存池或重用缓冲区，减少频繁的内存分配。

7. 构建与分发：优化用户体验

- 发布优化：除 LTO 外，尝试设置 codegen-units = 1 以进一步优化。
- 提供预编译 Wheels：利用 maturin 的 --universal2 和 --target 参数，为不同平台（macOS, Linux, Windows）提供预编译的 wheels，避免用户本地编译。

8. 总结与优先级建议

建议按以下优先级推进：

1. 性能分析（Profiling）：使用 criterion或 py-spy 等工具定位真正的热点，避免过早优化。
2. 优化 PyO3 绑定：重点检查热点函数的数据传递，实现零拷贝。
3. 下沉计算逻辑：将热点中的 Python 循环迁移到 Rust。
4. 调整构建配置：启用 LTO 等优化。
