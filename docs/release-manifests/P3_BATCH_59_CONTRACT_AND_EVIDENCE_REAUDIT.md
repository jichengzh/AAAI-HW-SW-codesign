# P3-59 合同与证据边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P7 启动授权。

本批从冻结 P2b 基线中确定性选取 12 个、且不在此前 470 个唯一已覆盖身份中的代码候选。所有候选完成 P2 metadata/bytes、当前内容 SHA-256、职责/调用/公开反证、许可证/来源和安全输出的只读审查。

| 处置 | 数量 | 主导责任 |
| --- | ---: | --- |
| 已安全改写 | 6 | capability/manifest、outlier、Stage4 import/cost selection、feedback 与 ranking contract |
| P4 | 5 | serializable cost bundle、evidence registry、native route、probe profile 与 measurement feedback evidence |
| P7 阻塞 | 1 | 会回显私有 probe/feature 标识、尚无公开安全改写和测试链的纯内存逻辑 |

第 7 项的前两次审阅同意它需安全改写，但独立复审使用了非协议 disposition；第三次只读裁决按协议确认它必须记为 P7 阻塞，而不能提前记为已改写。其当前没有公开等价 API 或测试，输出和异常会反射调用方的未净化标识；只有完成字段白名单/位置索引改写并以合成输入补足测试后，才可重新审阅。其余 5 项虽可能是纯 CPU 算法，但主导输入仍是外部测量、LUT、模型 bundle、probe run 或 feedback evidence，故归 P4；公开相邻选择逻辑不构成完整替代。

候选源码的项目受控部分继承 Apache-2.0；该许可不覆盖测量、LUT、模型 bundle、硬件遥测、AP/能耗/延迟证据及其派生物。后续 P4/P7 必须使用来源/哈希、路径无关 schema、匿名 fixture、受控输出根和失败关闭。本批没有运行候选代码、模型、数据、设备、子进程或网络任务；受限 records 与 HMAC key 不进入公开树，HMAC ledger 已重算。本批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **482/1,551**。
