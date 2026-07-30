"""验证 Index 类优化后方法的行为正确性。

对比优化前后的行为，确保：
1. intersection - 交集，保留首次出现顺序并去重
2. union - 并集，保留首次出现顺序并去重
3. unique - 去重，保留首次出现顺序
4. duplicated - 重复检测，支持 keep='first' / keep='last'
"""

from rspandas.indexes import Index


def test_intersection() -> None:
    """验证 intersection 方法。"""
    # 基本交集
    idx1 = Index([1, 2, 3, 4, 5])
    idx2 = Index([3, 4, 5, 6, 7])
    result = idx1.intersection(idx2)
    assert result.tolist() == [3, 4, 5], f"基本交集失败: {result.tolist()}"

    # 带重复元素的交集（应去重）
    idx1 = Index([1, 2, 2, 3, 3])
    idx2 = Index([2, 3, 3, 4])
    result = idx1.intersection(idx2)
    assert result.tolist() == [2, 3], f"带重复元素的交集失败: {result.tolist()}"

    # 空交集
    idx1 = Index([1, 2, 3])
    idx2 = Index([4, 5, 6])
    result = idx1.intersection(idx2)
    assert result.tolist() == [], f"空交集失败: {result.tolist()}"

    # 保留首次出现顺序
    idx1 = Index([3, 1, 2, 3])
    idx2 = Index([2, 3, 4])
    result = idx1.intersection(idx2)
    assert result.tolist() == [3, 2], f"保留顺序失败: {result.tolist()}"

    print("intersection 验证通过")


def test_union() -> None:
    """验证 union 方法。"""
    # 基本并集
    idx1 = Index([1, 2, 3])
    idx2 = Index([3, 4, 5])
    result = idx1.union(idx2)
    assert result.tolist() == [1, 2, 3, 4, 5], f"基本并集失败: {result.tolist()}"

    # 带重复元素的并集（应去重）
    idx1 = Index([1, 2, 2, 3])
    idx2 = Index([3, 3, 4, 4])
    result = idx1.union(idx2)
    assert result.tolist() == [1, 2, 3, 4], f"带重复元素的并集失败: {result.tolist()}"

    # 保留首次出现顺序
    idx1 = Index([3, 1, 2])
    idx2 = Index([2, 4, 1])
    result = idx1.union(idx2)
    assert result.tolist() == [3, 1, 2, 4], f"保留顺序失败: {result.tolist()}"

    # 空并集
    idx1 = Index([])
    idx2 = Index([1, 2])
    result = idx1.union(idx2)
    assert result.tolist() == [1, 2], f"空并集失败: {result.tolist()}"

    print("union 验证通过")


def test_unique() -> None:
    """验证 unique 方法。"""
    # 基本去重
    idx = Index([1, 2, 2, 3, 3, 3])
    result = idx.unique()
    assert result.tolist() == [1, 2, 3], f"基本去重失败: {result.tolist()}"

    # 保留首次出现顺序
    idx = Index([3, 1, 2, 1, 3])
    result = idx.unique()
    assert result.tolist() == [3, 1, 2], f"保留顺序失败: {result.tolist()}"

    # 空列表
    idx = Index([])
    result = idx.unique()
    assert result.tolist() == [], f"空列表去重失败: {result.tolist()}"

    # 无重复
    idx = Index([1, 2, 3])
    result = idx.unique()
    assert result.tolist() == [1, 2, 3], f"无重复去重失败: {result.tolist()}"

    print("unique 验证通过")


def test_duplicated() -> None:
    """验证 duplicated 方法。"""
    # keep='first'（默认）
    idx = Index([1, 2, 2, 3, 3, 3])
    result = idx.duplicated(keep="first")
    expected = [False, False, True, False, True, True]
    assert result == expected, f"keep='first' 失败: {result} != {expected}"

    # keep='last'
    idx = Index([1, 2, 2, 3, 3, 3])
    result = idx.duplicated(keep="last")
    expected = [False, True, False, True, True, False]
    assert result == expected, f"keep='last' 失败: {result} != {expected}"

    # 无重复
    idx = Index([1, 2, 3])
    result_first = idx.duplicated(keep="first")
    result_last = idx.duplicated(keep="last")
    assert result_first == [False, False, False], (
        f"无重复 keep='first' 失败: {result_first}"
    )
    assert result_last == [False, False, False], (
        f"无重复 keep='last' 失败: {result_last}"
    )

    # 空列表
    idx = Index([])
    result = idx.duplicated()
    assert result == [], f"空列表 duplicated 失败: {result}"

    # 全部相同
    idx = Index([5, 5, 5, 5])
    result_first = idx.duplicated(keep="first")
    result_last = idx.duplicated(keep="last")
    assert result_first == [False, True, True, True], (
        f"全部相同 keep='first' 失败: {result_first}"
    )
    assert result_last == [True, True, True, False], (
        f"全部相同 keep='last' 失败: {result_last}"
    )

    print("duplicated 验证通过")


def main() -> None:
    """主验证函数。"""
    print("=" * 60)
    print("验证 Index 类优化后的方法行为")
    print("=" * 60)

    test_intersection()
    test_union()
    test_unique()
    test_duplicated()

    print("=" * 60)
    print("所有验证通过！优化后的方法行为正确。")
    print("=" * 60)


if __name__ == "__main__":
    main()
