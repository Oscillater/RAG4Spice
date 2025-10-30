"""
任务数据模型模块

定义任务相关的数据结构和模型类。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


@dataclass
class Task:
    """HSPICE任务模型"""
    id: int
    title: str
    description: str
    visual_info: str = ""

    def __post_init__(self):
        """初始化后验证"""
        if self.id <= 0:
            raise ValueError("任务ID必须是正整数")

        if not self.title or not self.title.strip():
            raise ValueError("任务标题不能为空")

        if not self.title.endswith('.sp'):
            self.title = f"{self.title}.sp"

        if not self.description or not self.description.strip():
            raise ValueError("任务描述不能为空")

        # 清理字符串
        self.title = self.title.strip()
        self.description = self.description.strip()
        self.visual_info = self.visual_info.strip()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "visual_info": self.visual_info
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """从字典创建任务"""
        return cls(
            id=data.get("id", 1),
            title=data.get("title", "任务1.sp"),
            description=data.get("description", ""),
            visual_info=data.get("visual_info", "")
        )

    def copy(self) -> 'Task':
        """创建任务副本"""
        return Task.from_dict(self.to_dict())


@dataclass
class TaskAnalysis:
    """任务分析结果模型"""
    general_description: str
    tasks: List[Task] = field(default_factory=list)

    def __post_init__(self):
        """初始化后验证"""
        if not self.general_description or not self.general_description.strip():
            raise ValueError("总体描述不能为空")

        self.general_description = self.general_description.strip()

        # 验证任务列表
        for i, task in enumerate(self.tasks):
            if not isinstance(task, Task):
                raise ValueError(f"任务{i + 1}必须是Task类型")

    def add_task(self, task: Task) -> None:
        """添加任务"""
        if not isinstance(task, Task):
            raise ValueError("必须是Task类型")

        # 检查ID冲突
        existing_ids = [t.id for t in self.tasks]
        if task.id in existing_ids:
            # 生成新ID
            new_id = max(existing_ids) + 1 if existing_ids else 1
            task = Task(
                id=new_id,
                title=task.title,
                description=task.description,
                visual_info=task.visual_info
            )

        self.tasks.append(task)

    def remove_task(self, task_id: int) -> bool:
        """移除任务"""
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                del self.tasks[i]
                return True
        return False

    def get_task(self, task_id: int) -> Optional[Task]:
        """获取指定ID的任务"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def update_task(self, task_id: int, **kwargs) -> bool:
        """更新任务"""
        task = self.get_task(task_id)
        if task:
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            return True
        return False

    def get_next_task_id(self) -> int:
        """获取下一个可用的任务ID"""
        if not self.tasks:
            return 1
        return max(task.id for task in self.tasks) + 1

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "general_description": self.general_description,
            "tasks": [task.to_dict() for task in self.tasks]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskAnalysis':
        """从字典创建任务分析"""
        tasks = []
        for task_data in data.get("tasks", []):
            tasks.append(Task.from_dict(task_data))

        return cls(
            general_description=data.get("general_description", ""),
            tasks=tasks
        )

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'TaskAnalysis':
        """从JSON字符串创建任务分析"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class GenerationResult:
    """代码生成结果模型"""
    task_id: int
    title: str
    description: str
    analysis: str = ""
    hspice_code: str = ""
    error: str = ""
    success: bool = True

    def __post_init__(self):
        """初始化后设置"""
        if self.error:
            self.success = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "analysis": self.analysis,
            "hspice_code": self.hspice_code,
            "error": self.error,
            "success": self.success
        }


@dataclass
class SessionState:
    """会话状态模型"""
    task_analysis: Optional[TaskAnalysis] = None
    last_prompt: str = ""
    last_response: str = ""
    generation_results: List[GenerationResult] = field(default_factory=list)

    def clear_task_analysis(self) -> None:
        """清除任务分析"""
        self.task_analysis = None
        self.generation_results = []

    def set_task_analysis(self, analysis: TaskAnalysis) -> None:
        """设置任务分析"""
        self.task_analysis = analysis
        self.generation_results = []

    def add_generation_result(self, result: GenerationResult) -> None:
        """添加生成结果"""
        self.generation_results.append(result)

    def get_generation_result(self, task_id: int) -> Optional[GenerationResult]:
        """获取指定任务的生成结果"""
        for result in self.generation_results:
            if result.task_id == task_id:
                return result
        return None

    def has_successful_results(self) -> bool:
        """检查是否有成功的生成结果"""
        return any(result.success for result in self.generation_results)

    def get_successful_results(self) -> List[GenerationResult]:
        """获取所有成功的生成结果"""
        return [result for result in self.generation_results if result.success]


class TaskValidator:
    """任务验证器"""

    @staticmethod
    def validate_task_data(data: Dict[str, Any]) -> bool:
        """验证任务数据格式"""
        try:
            # 检查必需字段
            if "general_description" not in data or "tasks" not in data:
                return False

            # 验证总体描述
            if not isinstance(data["general_description"], str):
                return False

            # 验证任务列表
            if not isinstance(data["tasks"], list):
                return False

            # 验证每个任务
            for task_data in data["tasks"]:
                if not TaskValidator.validate_single_task_data(task_data):
                    return False

            return True

        except Exception:
            return False

    @staticmethod
    def validate_single_task_data(data: Dict[str, Any]) -> bool:
        """验证单个任务数据"""
        try:
            required_fields = ["id", "title", "description"]
            for field in required_fields:
                if field not in data:
                    return False

            # 验证ID
            if not isinstance(data["id"], int) or data["id"] <= 0:
                return False

            # 验证标题和描述
            if not isinstance(data["title"], str) or not isinstance(data["description"], str):
                return False

            if not data["title"].strip() or not data["description"].strip():
                return False

            return True

        except Exception:
            return False


# 便捷函数
def create_task(
    title: str,
    description: str,
    task_id: int = None,
    visual_info: str = ""
) -> Task:
    """
    创建新任务

    Args:
        title: 任务标题
        description: 任务描述
        task_id: 任务ID（可选）
        visual_info: 视觉信息

    Returns:
        Task: 创建的任务
    """
    if task_id is None:
        task_id = 1

    return Task(
        id=task_id,
        title=title,
        description=description,
        visual_info=visual_info
    )


def create_task_analysis(
    general_description: str,
    tasks: List[Task] = None
) -> TaskAnalysis:
    """
    创建任务分析

    Args:
        general_description: 总体描述
        tasks: 任务列表

    Returns:
        TaskAnalysis: 创建的任务分析
    """
    if tasks is None:
        tasks = []

    return TaskAnalysis(
        general_description=general_description,
        tasks=tasks
    )