"""
OCR处理模块

提供图像文字识别功能，支持多种图像格式和预处理。
"""

import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Optional, Tuple

from ..config.settings import get_tesseract_cmd


class OCRProcessor:
    """OCR处理器"""

    def __init__(self):
        """初始化OCR处理器"""
        # 配置Tesseract路径
        pytesseract.pytesseract.tesseract_cmd = get_tesseract_cmd()

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        预处理图像以提高OCR识别率

        Args:
            image: PIL图像对象

        Returns:
            np.ndarray: 预处理后的图像数组
        """
        # 转换为numpy数组
        img_array = np.array(image)

        # 转换为灰度图
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # 应用高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 自适应阈值处理
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # 形态学操作去除噪点
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def extract_text_from_image(
        self,
        image: Image.Image,
        preprocess: bool = True,
        language: str = 'chi_sim+eng'
    ) -> str:
        """
        从图像中提取文本

        Args:
            image: PIL图像对象
            preprocess: 是否进行预处理
            language: OCR语言设置

        Returns:
            str: 提取的文本

        Raises:
            RuntimeError: OCR处理失败
        """
        try:
            if preprocess:
                # 预处理图像
                processed_img = self.preprocess_image(image)
            else:
                # 直接转换格式
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    processed_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    processed_img = img_array

            # 配置Tesseract参数
            config = f'--oem 3 --psm 6 -l {language}'

            # 执行OCR
            text = pytesseract.image_to_string(processed_img, config=config)

            return text.strip()

        except Exception as e:
            raise RuntimeError(f"OCR处理失败: {str(e)}")

    def extract_text_with_confidence(
        self,
        image: Image.Image,
        preprocess: bool = True
    ) -> Tuple[str, float]:
        """
        从图像中提取文本并返回置信度

        Args:
            image: PIL图像对象
            preprocess: 是否进行预处理

        Returns:
            Tuple[str, float]: 提取的文本和平均置信度
        """
        try:
            if preprocess:
                processed_img = self.preprocess_image(image)
            else:
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    processed_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    processed_img = img_array

            # 获取详细数据
            data = pytesseract.image_to_data(
                processed_img,
                output_type=pytesseract.Output.DICT,
                lang='chi_sim+eng'
            )

            # 提取文本和置信度
            text_parts = []
            confidences = []

            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])

                if text and conf > 0:  # 过滤掉空文本和无效置信度
                    text_parts.append(text)
                    confidences.append(conf)

            full_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0.0

            return full_text, avg_confidence

        except Exception as e:
            raise RuntimeError(f"OCR处理失败: {str(e)}")


# 创建全局OCR处理器实例
ocr_processor = OCRProcessor()


def extract_text_from_image(
    image: Image.Image,
    preprocess: bool = True
) -> str:
    """
    便捷函数：从图像中提取文本

    Args:
        image: PIL图像对象
        preprocess: 是否进行预处理

    Returns:
        str: 提取的文本
    """
    return ocr_processor.extract_text_from_image(image, preprocess)