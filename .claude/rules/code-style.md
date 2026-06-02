# 代码规范

## 命名约定

- 变量/函数：`camelCase`
- 类：`PascalCase`
- 常量：`UPPER_SNAKE_CASE`

## Python 风格

- 缩进：4 空格
- 字符串：双引号 `""`
- 所有函数必须包含类型提示（Type Hints）
- 异常处理：禁止空 `except`，必须记录错误上下文

## 语言约定

- 代码、日志、Git commit 使用英文
- 代码注释和用户交互使用中文

## Git 提交格式

```
<type>(<scope>): <subject>
```

类型：`feat` / `fix` / `docs` / `style` / `refactor` / `test` / `chore`