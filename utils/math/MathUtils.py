class MathUtils:
    @staticmethod
    def convert_to_numbers(arr):
        """
        将数组中元素转为数字
        """
        result = []
        for element in arr:
            # 处理布尔值
            if element is True:
                result.append(1)
            elif element is False:
                result.append(0)
            # 处理 None
            elif element is None:
                result.append(0)  # 或保留 None，根据需求决定
            # 检查是否是数字
            elif isinstance(element, (int, float)):
                result.append(element)
            else:
                try:
                    # 尝试转换为数字
                    converted = float(element)
                    # 如果是整数形式，转换为整数
                    if converted.is_integer():
                        result.append(int(converted))
                    else:
                        result.append(converted)
                except (ValueError, TypeError):
                    # 无法转换则保留原始值
                    result.append(element)
        return result
