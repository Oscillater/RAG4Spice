#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG4Spice 向量数据库构建脚本

该脚本用于从HSPICE手册PDF构建向量数据库，支持增量更新和强制重建。
使用方法:
    python build_database.py                    # 构建数据库（如果不存在）
    python build_database.py --force-rebuild    # 强制重建数据库
    python build_database.py --pdf path/to/pdf # 指定PDF文件路径
"""

import argparse
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.database import build_database, db_manager
from src.config.settings import settings


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="构建RAG4Spice向量数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python build_database.py
  python build_database.py --force-rebuild
  python build_database.py --pdf custom_manual.pdf
  python build_database.py --info
        """
    )

    parser.add_argument(
        "--pdf",
        type=str,
        help=f"PDF文件路径 (默认: {settings.PDF_PATH})"
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="强制重建数据库（即使已存在）"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="显示数据库信息"
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="清除现有数据库"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细输出"
    )

    return parser.parse_args()


def show_database_info():
    """显示数据库信息"""
    print("📊 数据库信息")
    print("=" * 50)

    info = db_manager.get_database_info()

    for key, value in info.items():
        print(f"{key:20}: {value}")

    print("=" * 50)


def clear_database():
    """清除数据库"""
    print("🗑️  清除数据库")
    print("=" * 50)

    try:
        if db_manager.clear_database():
            print("✅ 数据库清除成功")
        else:
            print("⚠️  数据库不存在，无需清除")
    except Exception as e:
        print(f"❌ 数据库清除失败: {e}")
        return False

    print("=" * 50)
    return True


def build_vector_database(pdf_path: str, force_rebuild: bool, verbose: bool = False):
    """构建向量数据库"""
    print("🚀 构建RAG4Spice向量数据库")
    print("=" * 50)

    if verbose:
        print(f"配置信息:")
        print(f"  PDF路径: {pdf_path}")
        print(f"  数据库目录: {settings.PERSIST_DIRECTORY}")
        print(f"  嵌入模型: {settings.EMBEDDING_MODEL}")
        print(f"  块大小: {settings.CHUNK_SIZE}")
        print(f"  重叠大小: {settings.CHUNK_OVERLAP}")
        print("-" * 50)

    try:
        # 验证配置
        settings.validate()

        # 构建数据库
        db = build_database(pdf_path, force_rebuild)

        print("✅ 向量数据库构建完成！")

        if verbose:
            print(f"\n📈 构建统计:")
            info = db_manager.get_database_info()
            if "document_count" in info:
                print(f"  文档数量: {info['document_count']}")
            print(f"  数据库位置: {info['persist_directory']}")
            print(f"  嵌入模型: {info['embedding_model']}")

        return True

    except FileNotFoundError as e:
        print(f"❌ 文件错误: {e}")
        print("请确保PDF文件存在且路径正确")
        return False

    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        print("请检查环境变量和配置设置")
        return False

    except Exception as e:
        print(f"❌ 构建失败: {e}")
        if verbose:
            import traceback
            print("\n详细错误信息:")
            traceback.print_exc()
        return False

    finally:
        print("=" * 50)


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    try:
        # 显示数据库信息
        if args.info:
            show_database_info()
            return 0

        # 清除数据库
        if args.clear:
            if clear_database():
                return 0
            else:
                return 1

        # 构建数据库
        pdf_path = args.pdf or settings.PDF_PATH
        success = build_vector_database(
            pdf_path=pdf_path,
            force_rebuild=args.force_rebuild,
            verbose=args.verbose
        )

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        return 1

    except Exception as e:
        print(f"\n❌ 未预期的错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # 设置退出码
    sys.exit(main())